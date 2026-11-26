# MoE expert streaming: ubatch, exactness and waves, 2026-11-26

Evidence base for branch `feat/moe-stream-ubatch-waves` (fork feature, not upstream).

## Setup

- Model: OLMoE-1B-7B-0924-Instruct-Q4_K_M, 64 experts, 8 used per token.
- Host: `ai`, CPU only (AVX512 incl. VNNI), 16 threads.
- Functional runs only. No performance was measured for this document (see the end).
- Command: `llama-completion -no-cnv --temp 0 --seed 42 --no-warmup --no-display-prompt -n 48`,
  `-p` a word-list prompt, `-ub U`, streamed side `--moe-stream --moe-stream-cache Ns`.
- Plain and streamed sides always use the same `-ub` and the same `--repack` setting.

## Contract shipped

- The `n_ubatch` clamp is removed. The effective ubatch is the `-ub` requested, bounded only by `n_batch`.
- Default (no `--moe-stream-waves`): one pass per ubatch, bit-exact against a plain run with the same
  `--repack` setting.
- The single-pass remap loads every distinct expert of the ubatch at once, so `n_slots` must be at least
  that count (at most `n_expert`). The count is only known after the router runs.
- When a ubatch routes to more distinct experts than `n_slots`, the run stops deterministically before any
  eviction or load: exit code 134, empty stdout, one error message with the layer, the count, the slots,
  the recommended `--moe-stream-cache <Ns>` (`ceil(distinct*1.25)`, capped at `n_expert`) and the two
  alternatives (`-ub <= n_slots` for the exact path, `--moe-stream-waves` for speed with the caveat).
- `--moe-stream-waves` (default off) splits the experts of a ubatch into waves of `n_slots/2`. It is not
  bit-exact on CPU and prints one WARN line at startup.
- Comparison with master: master reduced the ubatch silently to `n_slots/n_expert_used`, which hid the
  case and never matched the requested `-ub`. The new behaviour lets the user decide and gives the full
  `-ub` once the cache is sized for it.

## Distinct experts per ubatch (OLMoE, Q4_K_M)

From the admission message, layer 0, cache 16s (chosen to force the abort), waves off:

| prompt | ubatch tokens | distinct experts |
| --- | --- | --- |
| P5 | 7 | 22 |
| P30 | 9 | 26 |
| P30 | 32 | 45 |
| P30 | 53 | 56 |
| P64 | 32 | 45 |
| P64 | 64 | 57 |
| P64 | 115 | 58 |

Reads: distinct grows with ubatch tokens and saturates near `n_expert`; with 64 experts a cache of
`64s` covers any prompt, a cache of `2*n_expert_used = 16s` covers decode (one token, 8 routes), and
anything between needs `-ub <= n_slots` or waves.

## Repack proof (four pairs, waves off, `-ub 64`, P30)

| pair | result |
| --- | --- |
| plain (repack on) vs plain `--no-repack` | DIFFER at byte 23 |
| plain `--no-repack` vs streamed `--no-repack` | identical byte for byte |
| plain (repack on) vs streamed (repack on) | DIFFER at byte 82 |

The plain run repacks the routed-expert weights into the CPU repack layout; the streamed cache cannot, its
buffer type must accept one expert slab per write (`set_tensor` of CPU_REPACK requires a single whole-tensor
write). The residual plain-vs-streamed difference is the repack kernel versus the stock kernel, the same
upstream divergence recorded in `.kilo/rules/upstream-sync.md` 4.2 (ref vs repack: 2.29e-05). The streamed
path is exact against a plain run of the same repack setting.

## Default-path matrix (waves off), functional only

60 cells: prompts P5/P30/P64 x `-ub` 9/64/512/2048 x caches 16s/20s/32s/40s/63s, same `-ub` on both
sides. `EQUAL` = byte-identical; `ABORT` = exit 134 with the admission message (distinct > slots).

Run with `--no-repack` on both sides, which is the contract setting (same repack regime):

| prompt | `-ub` | 16s | 20s | 32s | 40s | 63s |
| --- | --- | --- | --- | --- | --- | --- |
| P5 | 9/64/512/2048 | ABORT | ABORT | EQUAL | EQUAL | EQUAL |
| P30 | 9 | ABORT | ABORT | ABORT | EQUAL | EQUAL |
| P30 | 64/512/2048 | ABORT | ABORT | ABORT | ABORT | EQUAL |
| P64 | 9/64/512/2048 | ABORT | ABORT | ABORT | ABORT | EQUAL |

Result: **21 EQUAL, 39 ABORT, 0 DIFFER**. Every cell that fits the cache is byte-identical; the rest
stop with the admission message. P5 needs 32s because its 7-token ubatch routes to 22 experts, more than
16/20 slots.

With `--moe-stream-cache 64s` (one slot per expert, the size the message recommends for any prompt), all
12 prompt x `-ub` combinations are EQUAL, including P64 at `-ub 2048`.

In the default repack regime the same cells differ only where plain and streamed endpoints fall in the
repack-sensitive range (see the repack proof below), never because of the clamp or the ubatch.

## Waves

`ggml_mul_mat_id` on CPU builds a per-expert list of routed rows; the row count per expert decides the
kernel path and the accumulation order. Splitting the pairs by wave changes that count per expert, so an
expert computed in a wave is not bit-identical to the same expert in one pass. With `--moe-stream-waves`
the SOTA prompt path is available, at the cost of that drift. Keep it opt-in.

## Performance

Not measured. The numbers must be taken on a quiet machine (no other build or agent), with the load
average recorded before the run, and only after the user approves:

```
B=build/bin
M=OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
P=prompt.txt   # 115-token prompt
for ub in 64 512 2048; do
  for ns in 16 20 32 40 63; do
    for r in 1 2 3; do
      $B/llama-completion -m $M --no-warmup -no-cnv --temp 0 --seed 42 --no-display-prompt \
        -n 32 -ub $ub -p "$(cat $P)" --moe-stream --moe-stream-cache ${ns}s
    done
  done
done
```
