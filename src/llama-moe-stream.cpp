#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 32;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;
static const size_t MOE_STREAM_FILL_CHUNK   = 64ull*1024*1024; // bulk read size of the load-time fills
// the load-time fills allocate one staging buffer per thread; a large thread count would pin
// gigabytes of transient memory, so cap the fill parallelism independently of the load pool
static const int32_t MOE_STREAM_FILL_THREADS_MAX = 8;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // ReadFile with an explicit offset is a positional read, so the I/O workers do not share a
    // file pointer and can load experts in parallel
    const intptr_t os_handle = _get_osfhandle(file.file_id());
    if (os_handle == -1) {
        return nullptr;
    }
    uint8_t * p    = staging;
    size_t    left = len;
    ULONGLONG off  = (ULONGLONG) offs;
    while (left > 0) {
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffull);
        ov.OffsetHigh = (DWORD) (off >> 32);
        const DWORD chunk = (DWORD) std::min<size_t>(left, (size_t) 1 << 30);
        DWORD nread = 0;
        if (!ReadFile((HANDLE) os_handle, p, chunk, &nread, &ov) || nread == 0) {
            return nullptr;
        }
        p    += nread;
        off  += nread;
        left -= nread;
    }
    return staging;
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct, uint64_t ram_budget, int32_t pin_max)
        : n_slots(n_slots), ram_budget(ram_budget), pin_max(pin_max) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    if ((uint32_t) this->n_io_threads > MOE_STREAM_IO_THREADS_MAX) {
        LLAMA_LOG_WARN("%s: clamping MoE I/O threads from %d to %u\n", __func__, this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);
    }
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_claimed .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->expert_slot  .resize(n_expert, -1);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    file_paths = paths;

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

#ifdef _WIN32
    if (use_direct_io) {
        // llama_file::has_direct_io() is true on Windows but there is no O_DIRECT equivalent
        LLAMA_LOG_WARN("%s: O_DIRECT is not supported on Windows, falling back to buffered streaming reads\n", __func__);
        use_direct_io = false;
    }
#endif

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}

void llama_moe_stream::file_ranges(std::map<uint32_t, llama_mmap::ranges> & out) const {
    for (const auto & sl : layers) {
        if (!sl) {
            continue;
        }
        for (const auto & wt : sl->weights) {
            out[wt.file_idx].emplace_back(wt.offs, wt.offs + wt.nb_expert*sl->n_expert);
        }
    }
}

// runs n_jobs over a small thread pool, each thread with its own aligned staging buffer and thread
// index (used on Windows to give each fill thread its own file handles)
static bool llama_moe_stream_run_jobs(int32_t n_threads, size_t n_jobs, size_t nb_staging,
        const std::function<bool(size_t, uint8_t *, int32_t)> & job) {
    std::atomic<size_t> next(0);
    std::atomic<bool>   ok(true);

    std::vector<std::thread> pool;
    pool.reserve(n_threads);

    for (int32_t t = 0; t < n_threads; t++) {
        pool.emplace_back([&, t]() {
            uint8_t * staging = (uint8_t *) moe_aligned_alloc(nb_staging);
            if (staging == nullptr) {
                ok = false;
                return;
            }
            for (size_t i = next++; i < n_jobs && ok; i = next++) {
                if (!job(i, staging, t)) {
                    ok = false;
                }
            }
            moe_aligned_free(staging);
        });
    }
    for (auto & th : pool) {
        th.join();
    }

    return ok;
}

#ifdef _WIN32
// one file handle per fill thread: concurrent reads on a shared synchronous handle serialize
static std::vector<llama_files> llama_moe_stream_open_thread_files(const std::vector<std::string> & paths, int32_t n_threads) {
    std::vector<llama_files> out;
    out.reserve(n_threads);
    for (int32_t t = 0; t < n_threads; t++) {
        llama_files files;
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", false));
        }
        out.push_back(std::move(files));
    }
    return out;
}
#endif

// partition the expert cache: the low slots permanently hold the experts of the same ids, so they
// never pay a load again. The remaining slots form the dynamic pool used by the wave planner and
// the remap. A cache too small to spare slots for the pool leaves every slot dynamic.
void llama_moe_stream::pin_partition(uint32_t n_expert_used) {
    // the dynamic pool must hold one wave plus its preload, so two waves of n_expert_used
    const uint32_t n_dyn_min = std::max<uint32_t>(2*n_expert_used, 8);

    const int64_t t_start = ggml_time_us();

    struct job { llama_moe_stream_weight * wt; uint32_t n; };
    std::vector<job> jobs;

    uint32_t n_pinned = 0;
    uint32_t n_expert = 0;

    for (auto & sl_ptr : layers) {
        if (sl_ptr == nullptr) {
            continue;
        }
        auto & sl = *sl_ptr;

        // pin_max < 0: auto, half the cache stays dynamic
        // pin_max == 0: nothing pinned
        // pin_max > 0: at most pin_max pinned, never below n_dyn_min dynamic slots
        uint32_t n_dyn = sl.n_slots;
        if (pin_max < 0) {
            n_dyn = std::max<uint32_t>(n_dyn_min, sl.n_slots - sl.n_slots/2);
        } else if (pin_max > 0) {
            n_dyn = std::max<uint32_t>(n_dyn_min, sl.n_slots - std::min<uint32_t>(sl.n_slots, (uint32_t) pin_max));
        }

        if (sl.n_slots <= n_dyn) {
            sl.n_pinned = 0;
            continue;
        }

        sl.n_pinned = sl.n_slots - n_dyn;
        n_pinned    = std::max(n_pinned, sl.n_pinned);
        n_expert    = std::max(n_expert, sl.n_expert);

        for (auto & wt : sl.weights) {
            jobs.push_back({ &wt, sl.n_pinned });
        }

        for (uint32_t e = 0; e < sl.n_pinned; e++) {
            sl.slot_expert[e]   = (int32_t) e;
            sl.slot_state[e]    = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            sl.slot_last_use[e] = ++sl.use_counter;
            sl.expert_slot[e]   = (int32_t) e;
            sl.seen[e]          = 1;
        }
    }

    this->n_pinned = n_pinned;

    if (jobs.empty()) {
        if (pin_max == 0) {
            LLAMA_LOG_INFO("%s: expert pinning disabled, all %u slots dynamic\n", __func__, n_slots);
        } else {
            LLAMA_LOG_INFO("%s: expert cache too small for a pinned partition (%u dynamic slots needed), all slots dynamic\n",
                    __func__, n_dyn_min);
        }
        return;
    }

    const int32_t n_threads = std::min<int32_t>((int32_t) jobs.size(), std::max<int32_t>(std::min<int32_t>(n_io_threads, MOE_STREAM_FILL_THREADS_MAX), 1));
    const size_t fill_staging = std::max(MOE_STREAM_FILL_CHUNK, max_nb_expert) + 2*MOE_STREAM_DIRECT_ALIGN;

#ifdef _WIN32
    const auto thread_files = llama_moe_stream_open_thread_files(file_paths, n_threads);
#endif

    const bool ok = llama_moe_stream_run_jobs(n_threads, jobs.size(), fill_staging,
            [&](size_t i, uint8_t * staging, int32_t thread_idx) {
        const auto & j = jobs[i];
        const uint32_t g = std::max<uint32_t>(MOE_STREAM_FILL_CHUNK/j.wt->nb_expert, 1);
#ifdef _WIN32
        llama_file & wf = *thread_files[thread_idx][j.wt->file_idx];
#else
        GGML_UNUSED(thread_idx);
        llama_file & wf = *files[j.wt->file_idx];
#endif

        for (uint32_t e = 0; e < j.n; e += g) {
            const uint32_t n   = std::min(g, j.n - e);
            const size_t   nb  = (size_t) n*j.wt->nb_expert;
            const size_t   off = (size_t) e*j.wt->nb_expert;

            const uint8_t * src = llama_moe_stream_pread(wf, staging, nb, j.wt->offs + off, use_direct_io);
            if (src == nullptr) {
                return false;
            }
            ggml_backend_tensor_set(j.wt->cache, src, off, nb);
        }
        return true;
    });

    if (!ok) {
        throw std::runtime_error("MoE expert streaming: failed to load the pinned expert partition");
    }

    LLAMA_LOG_INFO("%s: %u of %u experts resident per layer, %u dynamic slots (%.2f s)\n",
            __func__, n_pinned, n_expert, n_slots - n_pinned, (ggml_time_us() - t_start)/1e6);
}

// mirror the experts the partition does not cover in pinned host memory, so their misses are a
// PCIe copy instead of a file read. Filled layer by layer; stops at the first layer that does not
// fit the budget, the remaining layers keep reading the file.
void llama_moe_stream::fill_host_mirror() {
    if (ram_budget == 0) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    struct job { llama_moe_stream_weight * wt; uint32_t first; uint32_t last; uint8_t * dst; };
    std::vector<job> jobs;

    size_t   left       = ram_budget;
    uint32_t n_layers   = 0;
    size_t   n_mirrored = 0;
    uint32_t n_device   = 0;
    uint32_t n_host     = 0;
    bool     stop       = false;

    for (auto & sl_ptr : layers) {
        if (sl_ptr == nullptr) {
            continue;
        }
        auto & sl = *sl_ptr;

        if (ggml_backend_buffer_is_host(sl.weights[0].cache->buffer)) {
            n_host++;
            continue; // the cache already lives in RAM, a mirror would duplicate it
        }
        n_device++;

        if (stop) {
            continue;
        }

        const uint32_t n_host_expert = sl.n_expert - sl.n_pinned;
        if (n_host_expert == 0) {
            continue;
        }

        size_t nb_layer = 0;
        for (const auto & wt : sl.weights) {
            nb_layer += (size_t) n_host_expert*wt.nb_expert;
        }
        if (nb_layer > left) {
            LLAMA_LOG_WARN("%s: host mirror budget too small for layer %d, mirroring stops here\n", __func__, sl.il);
            stop = true;
            continue;
        }

        // the mirror feeds device copies, so it must live in a pinned host buffer to keep the DMA
        // direct; a pageable or device buffer would defeat the point (and a device base is not
        // CPU-writable), so reject anything that is not a host buffer type
        const ggml_backend_buffer_type_t cache_buft = ggml_backend_buffer_get_type(sl.weights[0].cache->buffer);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(cache_buft);
        ggml_backend_dev_props props = {};
        if (dev != nullptr) {
            ggml_backend_dev_get_props(dev, &props);
        }
        ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!props.caps.host_buffer || buft == nullptr || !ggml_backend_buft_is_host(buft)) {
            LLAMA_LOG_WARN("%s: no pinned host buffer type for layer %d, host mirror disabled\n", __func__, sl.il);
            stop = true;
            continue;
        }

        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, nb_layer);
        if (buf == nullptr) {
            LLAMA_LOG_WARN("%s: host mirror allocation failed at layer %d, mirroring stops here\n", __func__, sl.il);
            stop = true;
            continue;
        }
        if (ggml_backend_buffer_get_type(buf) != buft) {
            // the pinned allocation fell back to pageable memory: keep streaming from disk instead
            ggml_backend_buffer_free(buf);
            LLAMA_LOG_WARN("%s: pinned host allocation fell back to pageable memory at layer %d, host mirror disabled\n",
                    __func__, sl.il);
            stop = true;
            continue;
        }

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        if (base == nullptr) {
            ggml_backend_buffer_free(buf);
            LLAMA_LOG_WARN("%s: host mirror buffer has no CPU base at layer %d, mirroring stops here\n", __func__, sl.il);
            stop = true;
            continue;
        }
        host_bufs.emplace_back(buf);

        for (auto & wt : sl.weights) {
            jobs.push_back({ &wt, sl.n_pinned, sl.n_expert, base });
            base += (size_t) n_host_expert*wt.nb_expert;
        }

        left       -= nb_layer;
        n_mirrored += nb_layer;
        n_layers++;
    }

    if (n_device == 0) {
        if (n_host > 0) {
            LLAMA_LOG_WARN("%s: expert cache is in host memory, host mirror is redundant -- disabled\n", __func__);
        }
        return;
    }

    if (jobs.empty()) {
        LLAMA_LOG_WARN("%s: no layer could be mirrored, host mirror disabled (see warnings above)\n", __func__);
        return;
    }

    const int32_t n_threads = std::min<int32_t>((int32_t) jobs.size(), std::max<int32_t>(std::min<int32_t>(n_io_threads, MOE_STREAM_FILL_THREADS_MAX), 1));
    const size_t fill_staging = std::max(MOE_STREAM_FILL_CHUNK, max_nb_expert) + 2*MOE_STREAM_DIRECT_ALIGN;

#ifdef _WIN32
    const auto thread_files = llama_moe_stream_open_thread_files(file_paths, n_threads);
#endif

    const bool ok = llama_moe_stream_run_jobs(n_threads, jobs.size(), fill_staging,
            [&](size_t i, uint8_t * staging, int32_t thread_idx) {
        const auto & j = jobs[i];
        const uint32_t g = std::max<uint32_t>(MOE_STREAM_FILL_CHUNK/j.wt->nb_expert, 1);
#ifdef _WIN32
        llama_file & wf = *thread_files[thread_idx][j.wt->file_idx];
#else
        GGML_UNUSED(thread_idx);
        llama_file & wf = *files[j.wt->file_idx];
#endif

        for (uint32_t e = j.first; e < j.last; e += g) {
            const uint32_t n   = std::min(g, j.last - e);
            const size_t   nb  = (size_t) n*j.wt->nb_expert;

            const uint8_t * src = llama_moe_stream_pread(wf, staging, nb, j.wt->offs + (size_t) e*j.wt->nb_expert, use_direct_io);
            if (src == nullptr) {
                return false;
            }
            memcpy(j.dst + (size_t) (e - j.first)*j.wt->nb_expert, src, nb);
        }
        return true;
    });

    if (!ok) {
        // nothing was published yet, so the workers still fall back to the file on every miss
        host_bufs.clear();
        throw std::runtime_error("MoE expert streaming: failed to fill the expert host mirror");
    }

    // publish only once every slab is in place, so a partial fill is never visible to the workers
    for (const auto & j : jobs) {
        j.wt->host       = j.dst;
        j.wt->host_first = j.first;
    }

    LLAMA_LOG_INFO("%s: expert host mirror = %.2f GiB over %u layers (%.2f s)\n",
            __func__, n_mirrored / 1024.0 / 1024.0 / 1024.0, n_layers, (ggml_time_us() - t_start)/1e6);
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop() {
    // pinned host staging when the device exposes it: pageable H2D copies are slow and serialize
    //   on CUDA/ROCm, which back-pressures the expert reads themselves. page-aligned anyway
    //   (Metal private buffers; O_DIRECT needs the extra head/tail slack for its aligned reads)
    ggml_backend_buffer_ptr staging_buf;
    uint8_t * staging = nullptr;
    const size_t staging_size = max_nb_expert + 4*MOE_STREAM_DIRECT_ALIGN;

    for (const auto & [buft, ctx] : ctxs) {
        GGML_UNUSED(ctx);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        ggml_backend_buffer_type_t host_buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft == nullptr) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(host_buft, staging_size);
        if (buf == nullptr) {
            continue;
        }
        staging_buf.reset(buf);
        staging = (uint8_t *) ggml_backend_buffer_get_base(buf);
        break;
    }
    if (staging == nullptr) {
        staging = (uint8_t *) moe_aligned_alloc(staging_size);
    } else {
        // O_DIRECT requires a 4096-aligned buffer; pinned allocations only guarantee 256 bytes
        staging = (uint8_t *) (((uintptr_t) staging + MOE_STREAM_DIRECT_ALIGN - 1) & ~(uintptr_t)(MOE_STREAM_DIRECT_ALIGN - 1));
    }
    GGML_ASSERT(staging != nullptr);

#ifdef _WIN32
    // each worker opens its own handles: concurrent reads on one synchronous file object serialize
    llama_files worker_files;
    try {
        for (const auto & path : file_paths) {
            worker_files.emplace_back(new llama_file(path.c_str(), "rb", false));
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mtx);
        load_failed = true;
        cv_done.notify_all();
        moe_aligned_free(staging);
        return;
    }
#endif

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        llama_moe_stream_work w = q_demand.front();
        q_demand.pop_front();

        auto & sl = *w.sl;
        if (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            sl.slot_claimed[w.slot]) {
            continue; // stale or duplicate item
        }
        sl.slot_claimed[w.slot] = 1;

        lk.unlock();

        bool ok = true;
        bool from_ram = true;
        for (const auto & wt : sl.weights) {
            const uint8_t * data = nullptr;
            if (wt.host != nullptr && (uint32_t) w.expert >= wt.host_first) {
                // pinned host mirror: the miss is a PCIe copy, not a file read
                data = wt.host + (size_t) (w.expert - wt.host_first)*wt.nb_expert;
            } else {
                from_ram = false;
#ifdef _WIN32
                llama_file & wf = *worker_files[wt.file_idx];
#else
                llama_file & wf = *files[wt.file_idx];
#endif
                data = llama_moe_stream_pread(wf, staging, wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
            }
            if (data == nullptr) {
                ok = false;
                break;
            }
            ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
        }

        lk.lock();

        sl.slot_claimed[w.slot] = 0;
        if (!ok) {
            load_failed = true;
        } else {
            sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            if (from_ram) {
                stats.n_load_ram++;
            } else {
                stats.n_load_disk++;
            }
        }
        cv_done.notify_all();
    }
    lk.unlock();

    if (!staging_buf) {
        moe_aligned_free(staging);
    }
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = sl.n_pinned; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot[sl.slot_expert[slot]] = -1;
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: expert loads = %" PRId64 " (ram = %" PRId64 ", disk = %" PRId64 ")\n",
            __func__, stats.n_load_ram + stats.n_load_disk, stats.n_load_ram, stats.n_load_disk);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    mgr->update_hotness_locked(*sl);

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const int32_t s = sl->expert_slot[e];
        if (s >= 0) {
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->q_demand.push_back({ sl, e, s, sl->slot_gen[s] });
                mgr->cv_work.notify_one();
                waited = true;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                mgr->cv_done.wait(lk);
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->q_demand.push_back({ sl, e, v, sl->slot_gen[v] });
            mgr->cv_work.notify_one();
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot[ids[i]];
        GGML_ASSERT(s >= 0);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

// bump the route hotness of the touched experts (sl.uniq) and halve every counter periodically so
// a formerly-hot expert ages out; shared by the decode remap and the prefill wave planner
void llama_moe_stream::update_hotness_locked(llama_moe_stream_layer & sl) {
    for (const int32_t e : sl.uniq) {
        sat_inc(sl.route_hotness[e]);
    }
    if (hot_decay_interval > 0 && stats.n_calls % hot_decay_interval == 0) {
        for (auto & sl2 : layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (sl.touched[e]) {
            continue;
        }
        sl.touched[e] = 1;
        if ((uint32_t) e < sl.n_pinned) {
            sl.expert_wave[e] = 0; // permanently resident, rides in the first wave at no slot cost
        } else {
            sl.uniq.push_back(e);
        }
    }

    update_hotness_locked(sl);

    GGML_ASSERT(sl.plan_capacity > 0);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = std::max<uint32_t>((uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity), 1);
    sl.plan_next_wave = 0;
}

// make wave w's expert slice (uniq[w*cap .. +count)) resident, waiting for its loads, and best-effort
// preload the next wave so its loads overlap this wave's compute. leaves sl.demand_slots = this wave's
// slots
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w) {
    const size_t first = (size_t) w*sl.plan_capacity;
    const size_t count = first < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - first) : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const int32_t s = sl.expert_slot[sl.uniq[i]];
        if (s >= 0) {
            sl.keep[s] = 1;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const int32_t s  = sl.expert_slot[e];
            if (s >= 0) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    q_demand.push_back({ &sl, e, s, sl.slot_gen[s] }); // promote to demand, wait for it
                    cv_work.notify_one();
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                q_demand.push_back({ &sl, e, v, sl.slot_gen[v] });
                cv_work.notify_one();
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot[e] >= 0) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            q_demand.push_back({ &sl, e, v, sl.slot_gen[v] });
            cv_work.notify_one();
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w;
// pairs whose expert belongs to another wave get -1 and are skipped by mul_mat_id
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] != (uint8_t) w) {
                out[i] = -1;
                continue;
            }
            const int32_t s = sl.expert_slot[e];
            GGML_ASSERT(s >= 0);
            GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            sl.slot_last_use[s] = ++sl.use_counter;
            out[i] = s;
        }
    }
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w); // make this wave resident, preload the next
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the pairs of other waves contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
