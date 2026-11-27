//
// test-moe-stream-waves - the wave grouping of the MoE expert streaming multi-pass prefill
//
// No model needed: the grouping is a pure function of the router ids. It checks the invariants the
// wave engine relies on: the waves cover every (token, route) pair exactly once, each token's
// routes stay inside one wave, and a wave writes -1 for the pairs of the other waves.
//

#include "llama-moe-stream.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { printf("%s:%d: FAIL %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

// one case: plan the waves of n_tok tokens routed to n_used experts of a 0..n_expert pool, then
// check the grouping and the per-wave slot emission. run_token_check requires every token's routes
// to stay inside one wave, which only holds when the case keeps a token's experts close in the
// first-use order (the engine relies on plan_capacity >= n_expert_used, not on this layout)
static void check_case(const char * name, const std::vector<int32_t> & ids, int64_t n_tok, uint32_t n_used,
        uint32_t n_expert, uint32_t plan_capacity, bool run_token_check) {
    const int failures_before = n_failed;

    std::vector<uint16_t> expert_wave;
    std::vector<int32_t> uniq;
    const uint32_t n_waves = llama_moe_stream_layer::plan_waves(ids.data(), n_tok, n_used, n_expert,
            plan_capacity, expert_wave, uniq);

    // the number of waves covers every distinct expert, plan_capacity at a time
    CHECK(n_waves == (uint32_t) ((uniq.size() + plan_capacity - 1)/plan_capacity));

    // every distinct touched expert is listed once, in first-use order
    {
        std::vector<uint8_t> seen(n_expert, 0);
        std::vector<int32_t> expect;
        for (const int32_t e : ids) {
            if (!seen[e]) {
                seen[e] = 1;
                expect.push_back(e);
            }
        }
        CHECK(expect == uniq);
        for (uint32_t e = 0; e < n_expert; e++) {
            CHECK((expert_wave[e] != 0xffff) == (seen[e] != 0));
        }
        for (size_t i = 0; i < uniq.size(); i++) {
            CHECK(expert_wave[uniq[i]] == (uint16_t) (i/plan_capacity));
            CHECK(expert_wave[uniq[i]] < n_waves);
        }
    }

    // a token's routes stay in one wave (the load-time invariant is n_expert_used <= plan_capacity)
    if (run_token_check) {
        for (int64_t t = 0; t < n_tok; t++) {
            const uint16_t w = expert_wave[ids[t*n_used]];
            for (uint32_t k = 1; k < n_used; k++) {
                CHECK(expert_wave[ids[t*n_used + k]] == w);
            }
        }
    }

    // the slot ids: expert e sits in cache slot e+1 so a 0 output is distinguishable from -1
    std::vector<int32_t> expert_slot(n_expert);
    for (uint32_t e = 0; e < n_expert; e++) {
        expert_slot[e] = (int32_t) e + 1;
    }

    std::vector<int32_t> out(ids.size());
    std::vector<int32_t> covered(ids.size(), 0);
    for (uint32_t w = 0; w < n_waves; w++) {
        std::fill(out.begin(), out.end(), -7);
        llama_moe_stream_layer::emit_wave_slots(expert_wave, ids.data(), expert_slot.data(), out.data(),
                (int32_t) w, n_used, n_tok);
        for (size_t i = 0; i < ids.size(); i++) {
            const int32_t e = ids[i];
            if (expert_wave[e] == w) {
                CHECK(out[i] == expert_slot[e]);
                covered[i]++;
            } else {
                CHECK(out[i] == -1);
            }
        }
    }

    // every pair is covered exactly once across the waves
    for (const int32_t c : covered) {
        CHECK(c == 1);
    }

    if (n_failed == failures_before) {
        printf("%s: %u tokens, %u used, %u experts, cap %u -> %u waves, %zu distinct: OK\n",
                name, (unsigned) n_tok, n_used, n_expert, plan_capacity, n_waves, uniq.size());
    } else {
        printf("%s: FAILED\n", name);
    }
}

int main() {
    // deterministic routing: each token takes n_used consecutive experts, wrapping around
    {
        const int64_t  n_tok = 30;
        const uint32_t n_used = 8, n_expert = 64;
        std::vector<int32_t> ids(n_tok*n_used);
        for (int64_t t = 0; t < n_tok; t++) {
            for (uint32_t k = 0; k < n_used; k++) {
                ids[t*n_used + k] = (int32_t) ((t*n_used + k) % n_expert);
            }
        }
        check_case("balanced", ids, n_tok, n_used, n_expert, 8, true);
        check_case("balanced, wide wave", ids, n_tok, n_used, n_expert, 32, true);
    }

    // shuffled routing, seed fixed: the grouping covers every pair once whatever the order
    {
        std::mt19937 rng(1234);
        const int64_t  n_tok = 24;
        const uint32_t n_used = 8, n_expert = 64;
        std::vector<int32_t> ids(n_tok*n_used);
        for (int64_t t = 0; t < n_tok; t++) {
            std::vector<int32_t> e(n_expert);
            for (uint32_t i = 0; i < n_expert; i++) {
                e[i] = (int32_t) i;
            }
            std::shuffle(e.begin(), e.end(), rng);
            for (uint32_t k = 0; k < n_used; k++) {
                ids[t*n_used + k] = e[k];
            }
        }
        check_case("shuffled", ids, n_tok, n_used, n_expert, 16, false);
    }

    // the decode shape: one token in a 2*n_expert_used cache keeps its routes in one wave
    {
        std::vector<int32_t> ids = { 5, 9, 2, 40, 17, 63, 0, 31 };
        check_case("one token", ids, 1, 8, 64, 8, true);
    }

    // more experts than a single byte can address: the wave index and the untouched marker no
    // longer collide (512 experts, 8 per wave -> 64 waves)
    {
        const int64_t  n_tok = 64;
        const uint32_t n_used = 8, n_expert = 512;
        std::vector<int32_t> ids(n_tok*n_used);
        for (int64_t t = 0; t < n_tok; t++) {
            for (uint32_t k = 0; k < n_used; k++) {
                ids[t*n_used + k] = (int32_t) ((t*n_used + k) % n_expert);
            }
        }
        check_case("512 experts", ids, n_tok, n_used, n_expert, 8, true);
    }

    if (n_failed != 0) {
        printf("test-moe-stream-waves: %d check(s) failed\n", n_failed);
        return 1;
    }

    printf("test-moe-stream-waves: all checks passed\n");
    return 0;
}
