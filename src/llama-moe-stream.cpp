#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <atomic>

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

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 16;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 32;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;
static const size_t   MOE_STREAM_STAGING_MAX        = 1ull*1024*1024*1024; // cap on the total staging of the I/O workers
static const uint64_t MOE_STREAM_RAM_FREE_PERCENT   = 50; // host mirror budget cap, in percent of available physical RAM
static const size_t   MOE_STREAM_FILL_CHUNK         = 64ull*1024*1024; // bulk read size of the load-time mirror fill
static const int32_t  MOE_STREAM_FILL_THREADS_MAX   = 8;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

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

// available physical RAM, 0 when unknown
static uint64_t llama_moe_stream_phys_avail() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }
    return status.ullAvailPhys;
#else
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES (MemFree)
    FILE * f = fopen("/proc/meminfo", "r");
    if (f != nullptr) {
        char line[256];
        uint64_t avail_kb = 0;
        while (fgets(line, sizeof(line), f) != nullptr) {
            if (sscanf(line, "MemAvailable: %" SCNu64 " kB", &avail_kb) == 1) {
                break;
            }
        }
        fclose(f);
        if (avail_kb > 0) {
            return avail_kb * 1024;
        }
    }
    const long pages     = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return (uint64_t) pages * (uint64_t) page_size;
#endif
}

static size_t moe_stream_staging_size(size_t max_nb_expert) {
    return max_nb_expert + 4*MOE_STREAM_DIRECT_ALIGN;
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

// map the exps tensors of one build_moe_ffn call to their streamed layer. looking at the pointers
// instead of the layer index also covers archs that evaluate several expert groups on one layer
// index, and archs that pass a different bid to the graph. a mix of streamed and non-streamed
// weights cannot be remapped safely, so abort instead of using expert ids on a resized cache.
llama_moe_stream_layer * llama_moe_stream::find_layer(
        const ggml_tensor * gate, const ggml_tensor * up,
        const ggml_tensor * down, const ggml_tensor * gate_up) const {
    size_t n_cached = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        for (const auto & sl : layers) {
            if (!sl) {
                continue;
            }
            for (const auto & wt : sl->weights) {
                if (wt.cache == t) {
                    n_cached++;
                    break;
                }
            }
        }
    }

    if (n_cached == 0) {
        return nullptr;
    }

    for (const auto & sl : layers) {
        if (sl && sl->matches(gate, up, down, gate_up)) {
            return sl.get();
        }
    }

    GGML_ABORT("MoE expert streaming: build_moe_ffn got a mix of streamed and non-streamed expert tensors "
               "(unsupported expert group or layer mapping)");
    return nullptr;
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct,
        uint64_t ram_budget, bool ram_auto)
        : n_slots(n_slots), ram_budget(ram_budget), ram_auto(ram_auto) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    if ((uint32_t) this->n_io_threads > MOE_STREAM_IO_THREADS_MAX) {
        LLAMA_LOG_WARN("%s: clamping MoE I/O threads from %d to %u\n", __func__, this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);
    }
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    trace         = std::getenv("LLAMA_MOE_STREAM_TRACE") != nullptr;
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
    ggml_set_name(cache, meta->name);
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
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    this->no_alloc = no_alloc;
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

        // the dummy buffer has size 0, report the size the cache tensors would need
        const size_t size = no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft) : ggml_backend_buffer_get_size(buf);
        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), size / 1024.0 / 1024.0, n_slots);
    }
}

void llama_moe_stream::clamp_io_staging() {
    const size_t staging_size = moe_stream_staging_size(max_nb_expert);
    const int32_t max_threads = (int32_t) std::max<size_t>(MOE_STREAM_STAGING_MAX / staging_size, 1);
    if (n_io_threads > max_threads) {
        LLAMA_LOG_WARN("%s: reducing MoE I/O threads from %d to %d (staging %.2f MiB per worker, %.2f MiB total, cap %.2f MiB)\n",
                __func__, n_io_threads, max_threads, staging_size / 1024.0 / 1024.0,
                staging_size * (size_t) max_threads / 1024.0 / 1024.0,
                MOE_STREAM_STAGING_MAX / 1024.0 / 1024.0);
        n_io_threads = max_threads;
    }

    LLAMA_LOG_INFO("%s: MoE I/O threads = %d, staging = %.2f MiB per worker, %.2f MiB total (cap %.2f MiB)\n",
            __func__, n_io_threads, staging_size / 1024.0 / 1024.0,
            staging_size * (size_t) n_io_threads / 1024.0 / 1024.0,
            MOE_STREAM_STAGING_MAX / 1024.0 / 1024.0);
}

// mirror an even share of the experts of every streamed layer in host memory, so their device
// misses are a PCIe copy instead of a file read. The budget is split by layer, not consumed as a
// prefix. Pinned host memory is used when the device exposes it, pageable host memory otherwise.
void llama_moe_stream::fill_host_mirror() {
    if (ram_budget == 0 && !ram_auto) {
        return;
    }

    const uint64_t ram_free = llama_moe_stream_phys_avail();
    const uint64_t ram_cap  = ram_free > 0 ? ram_free * MOE_STREAM_RAM_FREE_PERCENT / 100 : 0;

    if (ram_auto) {
        if (ram_cap == 0) {
            LLAMA_LOG_WARN("%s: cannot read available physical RAM, host mirror disabled\n", __func__);
            return;
        }
        ram_budget = ram_cap;
        LLAMA_LOG_INFO("%s: host mirror budget = %.2f GiB (%" PRIu64 "%% of available physical RAM)\n",
                __func__, ram_budget / 1024.0 / 1024.0 / 1024.0, MOE_STREAM_RAM_FREE_PERCENT);
    } else if (ram_cap > 0 && ram_budget > ram_cap) {
        // never allocate more than the same cap as auto, so a bad size cannot exhaust the host
        LLAMA_LOG_WARN("%s: host mirror budget %.2f GiB exceeds %" PRIu64 "%% of available physical RAM (%.2f GiB), clamping\n",
                __func__, ram_budget / 1024.0 / 1024.0 / 1024.0, MOE_STREAM_RAM_FREE_PERCENT,
                ram_cap / 1024.0 / 1024.0 / 1024.0);
        ram_budget = ram_cap;
    }
    if (ram_budget == 0) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    struct job {
        llama_moe_stream_weight * wt;
        uint32_t first;
        uint32_t last;
        uint8_t * dst;
    };
    std::vector<job> jobs;

    auto expert_bytes = [](const llama_moe_stream_layer & sl) {
        size_t nb = 0;
        for (const auto & wt : sl.weights) {
            nb += wt.nb_expert;
        }
        return nb;
    };

    uint32_t n_mirror_layers = 0;
    uint32_t n_device        = 0;
    uint32_t n_host          = 0;

    for (const auto & sl_ptr : layers) {
        if (sl_ptr == nullptr) {
            continue;
        }
        const auto & sl = *sl_ptr;

        if (ggml_backend_buffer_is_host(sl.weights[0].cache->buffer)) {
            n_host++;
            continue; // the cache already lives in RAM, a mirror would duplicate it
        }
        n_device++;
        if (expert_bytes(sl) > 0) {
            n_mirror_layers++;
        }
    }

    if (n_mirror_layers == 0) {
        if (n_host > 0 && n_device == 0) {
            LLAMA_LOG_WARN("%s: expert cache is in host memory, host mirror is redundant -- disabled\n", __func__);
        }
        return;
    }

    const uint64_t budget_per_layer = ram_budget / n_mirror_layers;

    uint32_t n_layers   = 0;
    size_t   n_mirrored = 0;
    bool     pinned     = true;

    for (auto & sl_ptr : layers) {
        if (sl_ptr == nullptr) {
            continue;
        }
        auto & sl = *sl_ptr;

        if (ggml_backend_buffer_is_host(sl.weights[0].cache->buffer)) {
            continue;
        }

        const size_t   nb_expert = expert_bytes(sl);
        const uint32_t n_expert  = sl.n_expert;
        if (nb_expert == 0) {
            continue;
        }

        const uint32_t n_host_expert = (uint32_t) std::min<uint64_t>(n_expert, budget_per_layer / nb_expert);
        if (n_host_expert == 0) {
            continue;
        }
        if (nb_expert > SIZE_MAX / n_host_expert) {
            LLAMA_LOG_WARN("%s: host mirror size overflows at layer %d, layer reads from file\n", __func__, sl.il);
            continue;
        }

        const size_t nb_layer = (size_t) n_host_expert*nb_expert;

        // prefer a pinned host buffer so the mirror feeds device copies directly; fall back to
        // pageable host memory when the device does not expose one
        const ggml_backend_buffer_type_t cache_buft = ggml_backend_buffer_get_type(sl.weights[0].cache->buffer);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(cache_buft);
        ggml_backend_dev_props props = {};
        if (dev != nullptr) {
            ggml_backend_dev_get_props(dev, &props);
        }
        ggml_backend_buffer_type_t buft = dev != nullptr && props.caps.host_buffer ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        bool layer_pinned = buft != nullptr && buft != ggml_backend_cpu_buffer_type() && ggml_backend_buft_is_host(buft);
        if (!layer_pinned) {
            buft = ggml_backend_cpu_buffer_type();
        }

        ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, nb_layer) };
        if (!buf) {
            LLAMA_LOG_WARN("%s: host mirror allocation failed at layer %d, layer reads from file\n", __func__, sl.il);
            continue;
        }

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf.get());
        if (base == nullptr || !ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buf.get()))) {
            LLAMA_LOG_WARN("%s: host mirror buffer has no host base at layer %d, layer reads from file\n", __func__, sl.il);
            continue;
        }
        // a pinned allocator can fall back to pageable memory; report the memory really used
        layer_pinned = layer_pinned && ggml_backend_buffer_get_type(buf.get()) == buft;
        pinned = pinned && layer_pinned;

        for (auto & wt : sl.weights) {
            jobs.push_back({ &wt, 0, n_host_expert, base });
            base += (size_t) n_host_expert*wt.nb_expert;
        }

        host_bufs.emplace_back(std::move(buf));
        n_mirrored += nb_layer;
        n_layers++;
    }

    if (jobs.empty()) {
        LLAMA_LOG_WARN("%s: no layer could be mirrored, host mirror disabled\n", __func__);
        host_bufs.clear();
        return;
    }

    const int32_t n_threads = std::max<int32_t>(1, std::min<int32_t>(
            std::min<int32_t>(n_io_threads, MOE_STREAM_FILL_THREADS_MAX), (int32_t) jobs.size()));
    const size_t fill_staging = std::max(MOE_STREAM_FILL_CHUNK, max_nb_expert) + 4*MOE_STREAM_DIRECT_ALIGN;

    std::atomic<size_t> next{0};
    std::atomic<bool>   failed{false};

    auto fill_worker = [&]() {
        llama_files thread_files;
        try {
            for (const auto & path : file_paths) {
                thread_files.emplace_back(new llama_file(path.c_str(), "rb", use_direct_io));
            }
        } catch (...) {
            failed = true;
            return;
        }

        uint8_t * staging = (uint8_t *) moe_aligned_alloc(fill_staging);
        if (staging == nullptr) {
            failed = true;
            return;
        }

        while (!failed) {
            const size_t i = next++;
            if (i >= jobs.size()) {
                break;
            }
            const job & j = jobs[i];
            const size_t   nb = j.wt->nb_expert;
            const uint32_t g  = (uint32_t) std::max<size_t>(MOE_STREAM_FILL_CHUNK/nb, 1);
            llama_file & wf = *thread_files[j.wt->file_idx];

            for (uint32_t e = j.first; e < j.last && !failed; e += g) {
                const uint32_t n  = std::min(g, j.last - e);
                const size_t   sz = (size_t) n*nb;

                const uint8_t * src = llama_moe_stream_pread(wf, staging, sz, j.wt->offs + (size_t) e*nb, use_direct_io);
                if (src == nullptr) {
                    failed = true;
                    break;
                }
                memcpy(j.dst + (size_t) (e - j.first)*nb, src, sz);
            }
        }
        moe_aligned_free(staging);
    };

    std::vector<std::thread> fill_threads;
    fill_threads.reserve(n_threads - 1);
    try {
        for (int32_t i = 0; i + 1 < n_threads; i++) {
            fill_threads.emplace_back(fill_worker);
        }
    } catch (const std::exception &) {
        // let the already started threads drain and exit before the vector is destroyed
        failed = true;
        next   = jobs.size();
        for (auto & t : fill_threads) {
            t.join();
        }
        host_bufs.clear();
        throw std::runtime_error("MoE expert streaming: failed to start the host mirror fill threads");
    }
    fill_worker();
    for (auto & t : fill_threads) {
        t.join();
    }

    if (failed) {
        // nothing was published yet, so the workers still fall back to the file on every miss
        host_bufs.clear();
        throw std::runtime_error("MoE expert streaming: failed to fill the expert host mirror");
    }

    // publish only once every slab is in place, so a partial fill is never visible to the workers
    for (const auto & j : jobs) {
        j.wt->host       = j.dst;
        j.wt->host_first = j.first;
        j.wt->host_last  = j.last;
    }

    LLAMA_LOG_INFO("%s: expert host mirror = %.2f GiB over %u layers (%s, %.2f s)\n",
            __func__, n_mirrored / 1024.0 / 1024.0 / 1024.0, n_layers, pinned ? "pinned" : "pageable",
            (ggml_time_us() - t_start)/1e6);
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    if (paths.empty()) {
        throw std::runtime_error("MoE expert streaming requires a file-based model");
    }
    for (const auto & path : paths) {
        if (path.empty() || path[0] == '(') {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
        std::error_code ec;
        const auto st = std::filesystem::status(std::filesystem::path(path), ec);
        if (ec || !std::filesystem::is_regular_file(st)) {
            throw std::runtime_error(format("MoE expert streaming requires a regular file, cannot stream '%s'", path.c_str()));
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

    for (const auto & sl : layers) {
        if (sl) {
            for (const auto & wt : sl->weights) {
                if (wt.file_idx >= files.size()) {
                    throw std::runtime_error(format("MoE expert streaming: tensor file index %u out of range (%zu files)", wt.file_idx, files.size()));
                }
            }
        }
    }

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty();
        for (const auto & file : files) {
            ok = ok && file->has_direct_io();
        }
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            for (const auto & file : files) {
                if (llama_moe_stream_pread(*file, probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) == nullptr) {
                    ok = false;
                    break;
                }
            }
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
    const size_t staging_size = moe_stream_staging_size(max_nb_expert);

    for (const auto & [buft, ctx] : ctxs) {
        GGML_UNUSED(ctx);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        ggml_backend_buffer_type_t host_buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft == nullptr || !ggml_backend_buft_is_host(host_buft)) {
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(host_buft, staging_size);
        if (buf == nullptr) {
            continue;
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        if (base == nullptr) {
            ggml_backend_buffer_free(buf);
            continue;
        }
        // O_DIRECT requires a 4096-aligned buffer; pinned allocations only guarantee 256 bytes
        base = (uint8_t *) (((uintptr_t) base + MOE_STREAM_DIRECT_ALIGN - 1) & ~(uintptr_t)(MOE_STREAM_DIRECT_ALIGN - 1));
        if (use_direct_io && !files.empty() &&
                llama_moe_stream_pread(*files.front(), base, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) == nullptr) {
            // device-mapped host memory (e.g. Vulkan) rejects O_DIRECT reads into it
            ggml_backend_buffer_free(buf);
            continue;
        }
        staging_buf.reset(buf);
        staging = base;
        break;
    }
    if (staging == nullptr) {
        staging = (uint8_t *) moe_aligned_alloc(staging_size);
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
            if (wt.host != nullptr && (uint32_t) w.expert >= wt.host_first && (uint32_t) w.expert < wt.host_last) {
                // static host mirror: the miss is a PCIe copy, not a file read
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
                stats.n_ram_hit++;
            } else {
                stats.n_ram_miss++;
            }
            if (debug) {
                LLAMA_LOG_DEBUG("%s: layer %d: expert %d -> slot %d\n", __func__, sl.il, w.expert, w.slot);
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

    for (uint32_t s = 0; s < sl.n_slots; s++) {
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
        stats.n_evictions++;
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_stream::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (no_alloc) {
        for (const auto & [buft, ctx_ptr] : ctxs) {
            res[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_ptr.get(), buft);
        }
    } else {
        for (const auto & buf : bufs) {
            res[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return res;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_dev_hit + stats.n_dev_miss;
    const int64_t n_loads   = stats.n_ram_hit + stats.n_ram_miss;
    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", device hits = %" PRId64 ", device misses = %" PRId64 ", hit rate = %.2f%%\n",
            __func__, n_calls, stats.n_dev_hit, stats.n_dev_miss,
            n_touched > 0 ? 100.0*stats.n_dev_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: ram hits = %" PRId64 ", ram misses = %" PRId64 ", loads = %" PRId64 " (disk = %" PRId64 "), ram hit rate = %.2f%%, evictions = %" PRId64 "\n",
            __func__, stats.n_ram_hit, stats.n_ram_miss, n_loads, stats.n_ram_miss,
            n_loads > 0 ? 100.0*stats.n_ram_hit/n_loads : 0.0, stats.n_evictions);
    LLAMA_LOG_INFO("%s: moe stream: device misses = %" PRId64 ", ram hits + ram misses = %" PRId64 ", load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.n_dev_miss, n_loads, stats.t_stall_us/1000.0, n_calls > 0 ? stats.t_stall_us/1000.0/n_calls : 0.0);
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

    mgr->n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        if (e < 0 || (uint32_t) e >= sl->n_expert) {
            GGML_ABORT("MoE expert streaming: layer %d got expert id %d (valid range 0..%u)",
                    sl->il, e, sl->n_expert - 1);
        }
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

    // routing trace (off by default); n = number of router ids, so a single-token decode call has n == n_expert_used
    if (mgr->trace) {
        LLAMA_LOG_INFO("%s: moe trace: layer=%d call=%" PRId64 " ids=%" PRId64 " distinct=%zu\n",
                __func__, sl->il, mgr->n_calls, n, sl->uniq.size());
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
            mgr->stats.n_dev_hit++;
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
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->q_demand.push_back({ sl, e, v, sl->slot_gen[v] });
            mgr->cv_work.notify_one();
            mgr->stats.n_dev_miss++;
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

// bump the route hotness of the touched experts (sl.uniq) and halve every counter periodically so
// a formerly-hot expert ages out
void llama_moe_stream::update_hotness_locked(llama_moe_stream_layer & sl) {
    for (const int32_t e : sl.uniq) {
        sat_inc(sl.route_hotness[e]);
    }
    if (hot_decay_interval > 0 && n_calls % hot_decay_interval == 0) {
        for (auto & sl2 : layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }
}
