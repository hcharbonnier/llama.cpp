#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Streaming of MoE routed expert weights from disk
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or unbuffered with
// --moe-stream-direct (O_DIRECT on POSIX, FILE_FLAG_NO_BUFFERING on Windows; bypasses the page
// cache; recommended when the model far exceeds RAM).
//
// --moe-stream-ram mirrors an even share of each streamed layer in host memory, pinned when the
// device exposes it. The device cache is dynamic, so the mirror may also hold resident experts.
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream;
struct llama_moe_stream_layer;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab

    const uint8_t * host       = nullptr; // static host mirror of a contiguous range of experts
    uint32_t        host_first = 0;       // first mirrored expert id
    uint32_t        host_last  = 0;       // one past the last mirrored expert id

    uint8_t * host_cache = nullptr; // hot policy: host cache base, n_host_slots slabs of this weight
};

// userdata of one wave's custom op (multi-pass prefill)
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t> slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t> slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t> slot_claimed;  // [n_slots] a worker owns the load
    std::vector<uint64_t> slot_gen;     // [n_slots] reservation generation
    std::vector<int64_t> slot_last_use; // [n_slots] LRU stamps
    std::vector<int32_t> expert_slot;   // [n_expert] RESIDENT or LOADING slot, -1 if absent

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    int64_t use_counter = 0;

    // hot host cache, guarded by mgr->mtx: n_host_slots slabs filled on admitted device misses
    uint32_t              n_host_slots = 0;
    std::vector<int32_t>  host_slot_expert;   // [n_host_slots] expert id or -1
    std::vector<uint8_t>  host_slot_state;    // [n_host_slots] llama_moe_stream_slot_state
    std::vector<uint32_t> host_slot_ref;      // [n_host_slots] claimed readers/writers, never evicted
    std::vector<uint64_t> host_slot_gen;      // [n_host_slots] reservation generation
    std::vector<int64_t>  host_slot_last_use; // [n_host_slots] LRU stamps
    std::vector<int32_t>  expert_host_slot;   // [n_expert] host slot, -1 if absent
    int64_t host_use_counter = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // consecutive passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave of each touched expert, 0xff = untouched
    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0; // stale unless it matches slot_gen[slot]

    bool     admit  = false; // hot policy: allow host insertion
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct,
            uint64_t ram_budget, bool ram_auto, bool ram_hot, bool ram_seed, bool ram_prefill);
    ~llama_moe_stream();

    llama_moe_stream_layer * find_layer(const ggml_tensor * gate, const ggml_tensor * up,
                                        const ggml_tensor * down, const ggml_tensor * gate_up) const;

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // cap n_io_threads so the per-worker staging buffers stay under a fixed total
    void clamp_io_staging();

    // open the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    // fill the static host mirror with an even share of the experts of every streamed layer
    void fill_host_mirror();

    // allocate the dynamic host cache (hot policy), empty at load time
    void setup_host_cache();

    // hot-seeded policy: fill the dynamic cache with the static mirror content at load time
    void fill_host_cache_seed();

    // mark multi-token decode calls (prompt evaluation) and return the previous state
    bool set_prefill(bool active) { return prefill_active.exchange(active, std::memory_order_relaxed); }

    // memory the cache buffers use, per buffer type; with no_alloc this is the projected size
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // print the streaming counters
    void print_stats() const;

    // a streamed model shares one expert cache across all its contexts, so only one context may
    // use it at a time; draft/MTP contexts built on the same model are refused too.
    // add_context returns 0 on success, else the number of contexts already active.
    uint32_t add_context() {
        std::lock_guard<std::mutex> lock(ctx_mtx);
        const uint32_t n_prev = n_contexts;
        if (n_prev == 0) {
            n_contexts = 1;
        }
        return n_prev;
    }

    void remove_context() {
        std::lock_guard<std::mutex> lock(ctx_mtx);
        GGML_ASSERT(n_contexts > 0);
        n_contexts--;
    }

    bool use_direct_io = false; // unbuffered streaming reads (--moe-stream-direct), no page cache

    uint64_t ram_budget  = 0; // host RAM tier budget in bytes
    bool     ram_auto    = false; // derive ram_budget from available physical RAM at load
    bool     ram_hot     = false; // --moe-stream-ram-policy hot: dynamic host cache, not a static mirror
    bool     ram_seed    = false; // hot-seeded: start from the static mirror content
    bool     ram_prefill = false; // --moe-stream-ram-prefill: admit during prompt evaluation too
    std::atomic<bool> prefill_active{false}; // set by the context around multi-token decode calls
    std::vector<ggml_backend_buffer_ptr> host_bufs;

    llama_files files; // privately opened GGUF files, same indices as the loader's

    // file paths, so each I/O worker can open its own handle (Windows synchronous I/O serializes
    // concurrent reads on a shared handle)
    std::vector<std::string> file_paths;

    size_t  max_nb_expert      = 0;
    uint32_t max_n_expert      = 0;
    int64_t hot_decay_interval = 0; // stream calls between route-hotness halvings (0 = no decay)
    int64_t n_calls            = 0;
    std::vector<int64_t> last_host_miss; // [n_expert] remap call of the last missed load, hot policy

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;
    bool no_alloc = false; // dummy buffers were attached, buffer sizes are only projected

    // single-context claim: guards against concurrent contexts sharing this expert cache
    std::mutex ctx_mtx;
    uint32_t   n_contexts = 0; // active contexts, guarded by ctx_mtx

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;
    bool trace = false; // LLAMA_MOE_STREAM_TRACE: log one line per remap call

    // counters, guarded by mtx: every touched expert is a device hit or miss, and every committed
    // load is a ram hit or a disk (ram miss) load
    struct {
        int64_t n_dev_hit   = 0; // touched experts already resident or loading
        int64_t n_dev_miss  = 0; // demand loads issued
        int64_t n_ram_hit   = 0; // loads served by the host mirror
        int64_t n_ram_miss  = 0; // loads read from the model file
        int64_t n_evictions = 0; // cache slots reused for another expert
        int64_t t_stall_us  = 0; // wait time in miss handling

        int64_t n_host_hit        = 0; // hot policy: loads served by the host cache
        int64_t n_host_miss       = 0; // hot policy: host lookups that read the file
        int64_t n_host_seeds      = 0; // hot policy: slabs published at load time by hot-seeded mode
        int64_t n_host_evictions  = 0; // hot policy: resident host slots reused for another expert
        int64_t n_host_insertions = 0; // hot policy: slabs published into the host cache
        int64_t t_host_insert_us  = 0; // hot policy: time spent copying slabs into the host cache

        int64_t n_wave_calls       = 0; // wave ids ops run (multi-pass prefill)
        int64_t n_waves_run        = 0; // non-empty waves staged
        int64_t n_preload_issued   = 0; // next-wave loads queued ahead of time
        int64_t n_preload_ready    = 0; // experts already resident when their wave started
        int64_t t_stall_wave_us    = 0; // wait time in wave staging
    } stats;

    // internals
    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);
    void update_hotness_locked(llama_moe_stream_layer & sl); // bump touched experts, decay periodically

    // multi-pass prefill internals (all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n);
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl,
            int32_t wave, bool host_admit);
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
            int32_t wave, uint32_t n_ids, int64_t n_tok);

    // hot host cache internals (all under mtx)
    int32_t pick_host_victim_locked(llama_moe_stream_layer & sl) const;
    void reserve_host_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);
};

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callback of one wave's id-remapping custom op (multi-pass prefill)
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata);
