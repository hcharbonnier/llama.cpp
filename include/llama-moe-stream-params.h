#pragma once

// fork-private MoE expert streaming configuration
// llama_model_params.moe_stream points here; a null pointer there means disabled

#include "llama.h"

#include <stdbool.h>
#include <stdint.h>

struct llama_moe_stream_params {
    bool     enabled;     // stream MoE routed expert weights from disk on demand
    uint32_t slots;       // expert cache slots per streamed layer (0 = auto)
    uint64_t budget;      // total expert cache byte budget, used when slots == 0 (0 = auto)
    int32_t  io_threads;  // expert load I/O threads (<= 0 = default)
    bool     direct;      // use direct I/O for expert reads (bypass the page cache)
    uint64_t ram;         // static host mirror byte budget for the experts the cache does not hold
    bool     ram_auto;    // derive the host mirror budget from available physical RAM
    bool     ram_hot;     // hot host tier: dynamic per-layer cache instead of a static mirror
    bool     ram_seed;    // hot host tier: seed the dynamic cache from the static mirror content
    bool     ram_prefill; // hot host tier: also admit experts during prompt evaluation
    bool     waves;       // multi-wave expert GEMMs: not bit-exact on CPU, see docs/moe-stream-waves-exactness-20261126.md
};

#ifdef __cplusplus
extern "C" {
#endif

// print MoE expert streaming statistics (no-op when streaming is not enabled)
LLAMA_API void llama_moe_stream_print_stats(const struct llama_model * model);

#ifdef __cplusplus
}
#endif
