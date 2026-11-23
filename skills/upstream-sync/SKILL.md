---
name: upstream-sync
description: Sync the fork master branch with ggml-org/llama.cpp master, replay moe-stream on the new mirror, promote master by fast-forward, run the verification invariants and publish with force-with-lease. Use for an upstream sync, when master must catch up with upstream, or when moe-stream must be replayed after the mirror moved.
---

# Sync the fork master with upstream

This skill is the operational companion of `.kilo/rules/upstream-sync.md`. Read that rule fully first: it is the source of truth for the branch model, the conflict policy and the invariants. This file orders the steps, adds the mandatory checkpoints and the report format, so an agent in a fresh session can run one sync end to end.

A sync needs an archive of the series before the rebase, otherwise the old series becomes unreachable once the old mirror is gone. Keep the pre-sync head and the previous mirror: `git branch -f archive/moe-stream-pre-sync-$(date +%Y%m%d) $(git rev-parse moe-stream)` runs before the rebase in step 3.

## Branch model

All commands run in `C:\dev\git\llama.cpp` (WSL: `/c/dev/git/llama.cpp`). A Git worktree shares the branches, so the same steps work from a worktree path; `patches/` only lands in the directory that runs the export.

| Branch | Role | Push rule |
| --- | --- | --- |
| `master` | Upstream plus the fork features, published | `--force-with-lease=master:<old-remote-hash>`, never bare `--force` |
| `upstream-mirror` | Pure upstream mirror, fetch and staging base | Never pushed, never edited by hand |
| `moe-stream` | Fork features, published | Replayed on `upstream-mirror` after each sync, never merged into |

History is linear (rebase), so every master update rewrites published history.

Also read `.kilo/rules/remote-machine-ai.md` (GPU host, `flock`) and `.kilo/rules/agent-workflow.md` (a history rewrite or a publication is a user checkpoint).

## Checkpoints (mandatory)

- Checkpoint A: before the push. Show the recorded hashes, the commit counts, the range-diff and the verification results, then wait for explicit human approval. The push rewrites the published history of `master`.
- Checkpoint B: before any GPU campaign (build on the GPU host, non-regression run, benchmark). Ask for explicit approval, then run it under `flock`.

Hard limits:

- Never fall back to bare `--force` or to an unleased push.
- Never delete `/tmp/radiance-gpu.lock`, never kill the process that holds it.
- When the lease fails, stop, close Checkpoint A and restart from step 1.
- Never `git reset --hard`, never edit `upstream-mirror`, never merge `master` into `moe-stream`.
- Never stage machine-specific untracked paths.

## Step 1 - Record the starting state

```sh
cd /c/dev/git/llama.cpp
git status --short --branch
git fetch origin
OLD_MIRROR=$(git rev-parse upstream-mirror)
OLD_REMOTE_MASTER=$(git rev-parse origin/master)
OLD_LOCAL_MASTER=$(git rev-parse master)
git rev-parse --short "$OLD_MIRROR" "$OLD_REMOTE_MASTER" "$OLD_LOCAL_MASTER"
```

`OLD_REMOTE_MASTER` equals `OLD_LOCAL_MASTER`. A mismatch is a blocker: explain it before touching anything. Keep both hashes for the lease and the report.

## Step 2 - Fetch upstream into the mirror

```sh
GIT_TERMINAL_PROMPT=0 git fetch upstream master:upstream-mirror
NEW_MIRROR=$(git rev-parse upstream-mirror)
echo "old=$OLD_MIRROR new=$NEW_MIRROR"
git rev-list --count "$OLD_MIRROR..$NEW_MIRROR"
git merge-base --is-ancestor "$OLD_MIRROR" "$NEW_MIRROR" && echo "mirror fast-forwarded"
```

The fetch stays silent when nothing changed, and silence proves nothing: compare the hashes. Equal hashes mean upstream did not move, stop and report it. `git rev-list --count "$OLD_MIRROR..$NEW_MIRROR"` of 0 confirms no upstream commit was lost. A failed ancestry test means the mirror moved backwards and needs a human decision before any rebase.

## Step 3 - Replay moe-stream on the new mirror

```sh
git switch moe-stream
git rev-list --count upstream-mirror..moe-stream
git rebase upstream-mirror
git rev-list --count upstream-mirror..moe-stream
git range-diff upstream-mirror...moe-stream
```

- Keep the two counts: both must match, one entry per fork commit, no squash, no `--skip`.
- Resolve conflicts commit by commit, see the conflict policy below.
- Replay a subset only with `git rebase --onto upstream-mirror <old-base> moe-stream`.
- Keep the pre-replay head: `git branch -f archive/moe-stream-pre-sync-$(date +%Y%m%d) $(git rev-parse moe-stream)` before the rebase.
- A commit that no longer builds after the replay stops the sync.
- `git range-diff upstream-mirror...moe-stream` run after the rebase lists every commit as new (`-:  ---->`), since the mirror itself moved. The meaningful comparison is the old series against the new one: `git range-diff <old-mirror>..<old-moe-stream> upstream-mirror..moe-stream`, and `19/19` equal lines rather than `-:` lines. Record which form was used.

## Step 4 - Promote master by fast-forward

```sh
git switch master
git merge --ff-only moe-stream
git rev-parse master moe-stream
git rev-list --count upstream-mirror..master
git rev-list --count upstream-mirror..master --merges
git log --oneline --graph -n 20 upstream-mirror..master
```

`master` and `moe-stream` share one hash, the merge count is 0, and the graph has no merge commit.

## Step 5 - Build and verify

Every point is mandatory and every point runs before the push:

- Build CPU, in addition to the default build.
- Build CUDA with `-DGGML_CUDA=ON`. A green CPU build alone is not a verification.
- No CLI option without consumer: every `--moe-stream*` option parsed in `common/arg.cpp` is read in `src/`; a parsed but never-read option fails the sync.
- No dead code shipped silently: nothing the sync left unreachable or unused.
- No declared-but-undefined symbol: the link passes on both targets, check the CUDA link for symbols such as `ggml_cuda_zero_skipped_mul_mat_id`.
- `ctest` green on the CPU build.

```sh
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release -j
ctest --test-dir build-cpu --output-on-failure

cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j
```

The local builds do not approve the GPU. The CUDA run follows `.kilo/rules/remote-machine-ai.md`: copy the repository state to the GPU host, never reuse an untracked working tree there, and run every GPU command under `flock -w 7200 /tmp/radiance-gpu.lock`.

Non-regression, mandatory, on at least one MoE model:

- Byte-for-byte equality of `llama-completion` output, streaming vs non-streaming, in CPU and in CUDA.
- The same equality with waves (an ubatch that touches more experts than slots, `-ub 512` and `2048`, prompts of 256 and 150 tokens) and without waves.
- The dense check (`SmolLM-135M`) and the MoE without streaming, as the previous sync did.
- `llama-cli` is an upstream chat client: measure and compare with `llama-completion`, not `llama-cli`.

Record an md5 of each output pair, not a bare "PASS".

## Step 6 - Checkpoint A: approval to publish

Show the user, then wait:

- `OLD_MIRROR` -> `NEW_MIRROR`, `OLD_REMOTE_MASTER` -> the new `master`,
- `upstream-mirror..moe-stream` count before and after, the range-diff result,
- build CPU, build CUDA, ctest results,
- non-regression model, configurations, md5 results,
- `git status --short --branch` and `git status --short --untracked-files=all`.

Nothing untracked or machine-specific may be staged. No approval means no push: stop and report the current state.

## Step 7 - Push

Run this step from the Windows git in `C:\dev\git\llama.cpp`, either natively in PowerShell or through `powershell.exe -NoProfile -Command ...`. Do not run it with the WSL git: it has no credential helper and fails with `could not read Username for 'https://github.com'`. The alternative is to configure a credential helper in WSL first.

```powershell
Set-Location C:\dev\git\llama.cpp
git fetch origin
$old = git rev-parse origin/master
git push --force-with-lease=master:$old origin master
git push -u origin moe-stream
```

- `$old` must be `OLD_REMOTE_MASTER` shown at Checkpoint A. Read the hash from a variable, never type it by hand.
- If the `git fetch origin` moved `origin/master`, go back to step 2 and restart the sync.
- If the push is rejected, the lease failed and the remote moved: nothing was written, restart from step 1, never from a bare `--force`.
- The two pushes are not atomic. If `master` is pushed and `moe-stream` is not, run Checkpoint A again for the second push.

## Step 8 - Post-push checks

```sh
git fetch origin
git ls-remote --heads origin master
git rev-parse master
git ls-remote --heads origin moe-stream
git rev-parse moe-stream
git status --short --branch
```

Both remote hashes equal their local heads, and the status shows no ahead or behind.

## Conflict policy

- Upstream and the fork implement the same feature differently: keep the upstream version.
- Fork-only behavior: keep the local version, adapted to the upstream API.
- Never resolve with "take the upstream file". Inventory the local hunks first and justify every dropped one.
- `skills/upstream-sync/` sits on a tracked path: keep the upstream hunks, re-apply the fork steps on top, and check that the replay drops neither a fork skill nor an upstream one.

## Artifacts

```sh
git format-patch upstream-mirror..moe-stream -o patches/
```

`patches/` is an untracked output for sharing and archiving, never the source of truth. The source of truth is the pair (`upstream-mirror`, `moe-stream`). Never edit `AGENTS.md`, which belongs to upstream.

## Report

The sync report contains at least:

| Field | Value |
| --- | --- |
| `upstream-mirror` | old hash -> new hash |
| `master` published | old remote hash -> new hash |
| Commit counts | `upstream-mirror..moe-stream` before/after, `old-mirror..new-mirror` integrated |
| Range-diff | n/n commits preserved, or the explained deltas |
| Build CPU | result |
| Build CUDA | `-DGGML_CUDA=ON`, link result |
| ctest | pass/fail count |
| Non-regression | model, configurations, byte-for-byte md5, CPU and CUDA, with and without waves |
| Post-push | `ls-remote` equals the local head for `master` and `moe-stream`, clean status |
| Archive | pre-sync branch kept for the old head, if any |

Record both hashes so the check stays reproducible. Close with what still fails or is untested, including the backends the fork does not verify (Metal, Vulkan, SYCL, ...).
