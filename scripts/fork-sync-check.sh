#!/usr/bin/env bash
#
# fork-sync-check.sh - non-regression and sync orchestration for this fork.
#
# Companion to skills/upstream-sync/SKILL.md and .kilo/rules/upstream-sync.md.
# It automates the repetitive parts of a sync: enabling rerere, fetching
# upstream into the mirror with before/after hash comparison, building CPU and
# CUDA, and the byte-for-byte non-regression of `llama-completion`.
#
# Usage:
#   scripts/fork-sync-check.sh [SUBCOMMAND] [OPTIONS]
#
# Subcommands:
#   check        (default) run the non-regression suite and ctest
#   build        build the CPU tree (Ninja) and the CUDA tree (-DGGML_CUDA=ON)
#   sync-prep    enable rerere, fetch upstream into upstream-mirror, print
#                the range-diff command. It never pushes and never rewrites.
#   remote-cuda  internal: run only the CUDA pair check. Copied and launched
#                on the GPU host by `check --remote`, under flock.
#
# Common options (flags override the matching environment variable):
#   --model PATH          MoE .gguf model                  (env MODEL)
#   --build-cpu DIR       local CPU build dir              (env BUILD_CPU=build-cpu)
#   --build-cuda DIR      local CUDA build dir             (env BUILD_CUDA=build-cuda)
#   --binary NAME         inference binary                 (env BINARY=llama-completion)
#   --prompt TEXT         fixed prompt                     (env PROMPT)
#   --n-predict N         fixed token budget               (env N_PREDICT=64)
#   --seed N              sampling seed                    (env SEED=42)
#   --temp N              sampling temperature             (env TEMP=0)
#   --ctx N               context size, empty = default    (env CTX)
#   --ubatch N            ubatch size, empty = default     (env UBATCH)
#   --stream-args TEXT    flags that enable streaming      (env STREAM_ARGS=--moe-stream)
#   --plain-args TEXT     flags for the non-streaming run  (env PLAIN_ARGS)
#   --repo-root DIR       repository root                  (env REPO_ROOT; default: script parent)
#   --out-dir DIR         output directory                 (env OUT_DIR; default: temp dir)
#   --ctest-regex RE      ctest selection                  (env CTEST_REGEX)
#   --skip-cpu            skip the local CPU pair          (env SKIP_CPU=0)
#   --skip-cuda           skip the CUDA pair               (env SKIP_CUDA=0)
#   --skip-ctest          skip ctest                       (env SKIP_CTEST=0)
#
# check --remote options (GPU parts over SSH, per .kilo/rules/remote-machine-ai.md):
#   --remote              run the CUDA pair on the GPU host
#   --remote-user USER    SSH user                         (env REMOTE_USER=agent-ia)
#   --remote-host HOST    SSH host                         (env REMOTE_HOST=192.168.0.156)
#   --remote-dir DIR      remote repository root           (env REMOTE_DIR=/opt/agent-ia/llama.cpp)
#   --remote-model PATH   remote MoE model, required       (env REMOTE_MODEL)
#   --remote-build-dir D  remote CUDA build dir            (env REMOTE_BUILD_CUDA=build-cuda)
#   --remote-script PATH  where to copy this script        (env REMOTE_SCRIPT=/tmp/fork-sync-check.sh)
#   --remote-out-dir DIR  remote output directory           (env REMOTE_OUT_DIR=/tmp/fork-sync-out)
#   --gpu-lock PATH       lock file                        (env GPU_LOCK=/tmp/radiance-gpu.lock)
#   --gpu-lock-timeout N  flock wait in seconds            (env GPU_LOCK_TIMEOUT=7200)
#
# sync-prep options:
#   --upstream REMOTE     upstream git remote             (env UPSTREAM_REMOTE=upstream)
#   --mirror BRANCH       mirror branch                  (env MIRROR_BRANCH=upstream-mirror)
#   --fork BRANCH         fork branch                    (env FORK_BRANCH=moe-stream)
#   --archive BRANCH      pre-sync archive branch name    (env ARCHIVE_BRANCH, default: archive/<fork>-pre-sync-<date>)
#
# The suite compares the streaming run against the non-streaming run on the
# same backend with `cmp`, so the comparison is exact. It records the md5 of
# each pair. It exits non-zero when any check fails.
#
# Examples:
#   scripts/fork-sync-check.sh build
#   scripts/fork-sync-check.sh check --model models/moe.gguf
#   scripts/fork-sync-check.sh check --model models/moe.gguf --remote \
#       --remote-model /opt/agent-ia/models/moe.gguf
#   scripts/fork-sync-check.sh sync-prep
#
# ASCII only. LF line endings. Requires bash >= 4.4.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$SCRIPT_NAME"
SUBCOMMAND="check"

MODEL="${MODEL:-}"
BUILD_CPU="${BUILD_CPU:-build-cpu}"
BUILD_CUDA="${BUILD_CUDA:-build-cuda}"
BINARY="${BINARY:-llama-completion}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog.}"
N_PREDICT="${N_PREDICT:-64}"
SEED="${SEED:-42}"
TEMP="${TEMP:-0}"
CTX="${CTX:-}"
UBATCH="${UBATCH:-}"
STREAM_ARGS="${STREAM_ARGS:---moe-stream}"
PLAIN_ARGS="${PLAIN_ARGS:-}"
REPO_ROOT="${REPO_ROOT:-}"
OUT_DIR="${OUT_DIR:-}"
CTEST_REGEX="${CTEST_REGEX:-^(test-arg-parser|test-llama-mmap)$}"
JOBS="${JOBS:-}"
SKIP_CPU="${SKIP_CPU:-0}"
SKIP_CUDA="${SKIP_CUDA:-0}"
SKIP_CTEST="${SKIP_CTEST:-0}"

REMOTE=0
REMOTE_USER="${REMOTE_USER:-agent-ia}"
REMOTE_HOST="${REMOTE_HOST:-192.168.0.156}"
REMOTE_DIR="${REMOTE_DIR:-/opt/agent-ia/llama.cpp}"
REMOTE_MODEL="${REMOTE_MODEL:-}"
REMOTE_BUILD_CUDA="${REMOTE_BUILD_CUDA:-build-cuda}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-/tmp/fork-sync-check.sh}"
REMOTE_OUT_DIR="${REMOTE_OUT_DIR:-/tmp/fork-sync-out}"
GPU_LOCK="${GPU_LOCK:-/tmp/radiance-gpu.lock}"
GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-7200}"

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
MIRROR_BRANCH="${MIRROR_BRANCH:-upstream-mirror}"
FORK_BRANCH="${FORK_BRANCH:-moe-stream}"
ARCHIVE_BRANCH="${ARCHIVE_BRANCH:-}"

FAILURES=0
RESULTS=()

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$SCRIPT_PATH"
}

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

record() {
    local status="$1" desc="$2"
    RESULTS+=("$status  $desc")
    if [ "$status" = "FAIL" ]; then
        FAILURES=$((FAILURES + 1))
    fi
    printf '%-4s %s\n' "$status" "$desc"
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

resolve_repo_root() {
    if [ -z "$REPO_ROOT" ]; then
        local script_dir
        script_dir="$(cd "$(dirname "$0")" && pwd)"
        REPO_ROOT="$(cd "$script_dir/.." && pwd)"
    fi
    if [ ! -d "$REPO_ROOT" ]; then
        die "repository root not found: $REPO_ROOT"
    fi
    REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
}

is_value_opt() {
    case "$1" in
        --model|--build-cpu|--build-cuda|--binary|--prompt|--n-predict|--seed|\
        --temp|--ctx|--ubatch|--stream-args|--plain-args|--repo-root|--out-dir|\
        --ctest-regex|--jobs|-j|--remote-user|--remote-host|--remote-dir|\
        --remote-model|--remote-build-dir|--remote-script|--remote-out-dir|\
        --gpu-lock|--gpu-lock-timeout|--upstream|--mirror|--fork|--archive)
            return 0 ;;
        *) return 1 ;;
    esac
}

parse_args() {
    local opt val has_eq consume
    while [ $# -gt 0 ]; do
        opt="$1"; val=""; has_eq=0; consume=1
        case "$opt" in
            --*=*) val="${opt#*=}"; opt="${opt%%=*}"; has_eq=1 ;;
        esac
        if is_value_opt "$opt"; then
            if [ "$has_eq" -eq 0 ]; then
                if [ $# -lt 2 ]; then
                    die "option $opt requires a value"
                fi
                val="$2"
                consume=2
            fi
        fi
        case "$opt" in
            --model)              MODEL="$val" ;;
            --build-cpu)          BUILD_CPU="$val" ;;
            --build-cuda)         BUILD_CUDA="$val" ;;
            --binary)             BINARY="$val" ;;
            --prompt)             PROMPT="$val" ;;
            --n-predict)          N_PREDICT="$val" ;;
            --seed)               SEED="$val" ;;
            --temp)               TEMP="$val" ;;
            --ctx)                CTX="$val" ;;
            --ubatch)             UBATCH="$val" ;;
            --stream-args)        STREAM_ARGS="$val" ;;
            --plain-args)         PLAIN_ARGS="$val" ;;
            --repo-root)          REPO_ROOT="$val" ;;
            --out-dir)            OUT_DIR="$val" ;;
            --ctest-regex)        CTEST_REGEX="$val" ;;
            --jobs|-j)            JOBS="$val" ;;
            --skip-cpu)           SKIP_CPU=1 ;;
            --skip-cuda)          SKIP_CUDA=1 ;;
            --skip-ctest)         SKIP_CTEST=1 ;;
            --remote)             REMOTE=1 ;;
            --remote-user)        REMOTE_USER="$val" ;;
            --remote-host)        REMOTE_HOST="$val" ;;
            --remote-dir)         REMOTE_DIR="$val" ;;
            --remote-model)       REMOTE_MODEL="$val" ;;
            --remote-build-dir)   REMOTE_BUILD_CUDA="$val" ;;
            --remote-script)      REMOTE_SCRIPT="$val" ;;
            --remote-out-dir)     REMOTE_OUT_DIR="$val" ;;
            --gpu-lock)           GPU_LOCK="$val" ;;
            --gpu-lock-timeout)   GPU_LOCK_TIMEOUT="$val" ;;
            --upstream)           UPSTREAM_REMOTE="$val" ;;
            --mirror)             MIRROR_BRANCH="$val" ;;
            --fork)               FORK_BRANCH="$val" ;;
            --archive)            ARCHIVE_BRANCH="$val" ;;
            -h|--help)            usage; exit 0 ;;
            -*)                   die "unknown option: $opt" ;;
            *)                    die "unexpected argument: $opt" ;;
        esac
        shift "$consume"
    done

    if [ -z "$JOBS" ]; then
        if command -v nproc >/dev/null 2>&1; then
            JOBS="$(nproc)"
        else
            JOBS="4"
        fi
    fi
}

split_list() {
    local -n ref="$1"
    ref=()
    if [ -n "$2" ]; then
        read -r -a ref <<< "$2"
    fi
}

run_once() {
    local build_dir="$1" out="$2"
    shift 2
    build_dir="$(abs_path "$build_dir")"
    local bin="$build_dir/bin/$BINARY"
    if [ ! -x "$bin" ]; then
        warn "missing binary: $bin"
        return 1
    fi

    local -a cmd=("$bin" -m "$MODEL" -p "$PROMPT" -n "$N_PREDICT" -s "$SEED"
                  --temp "$TEMP" --top-k 1 --top-p 1.0 --no-display-prompt -no-cnv)
    if [ -n "$CTX" ]; then
        cmd+=(-c "$CTX")
    fi
    if [ -n "$UBATCH" ]; then
        cmd+=(-ub "$UBATCH")
    fi
    cmd+=("$@")

    mkdir -p "$(dirname "$out")"
    ( cd "$REPO_ROOT" && "${cmd[@]}" ) >"$out" 2>"$out.err"
}

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        printf 'no-md5-tool'
    fi
}

# run_pair LABEL BUILD_DIR [extra args...]
run_pair() {
    local label="$1" build_dir="$2"
    shift 2
    local dir="$OUT_DIR"
    mkdir -p "$dir"

    local plain="$dir/$label.plain.txt"
    local stream="$dir/$label.stream.txt"

    log "-- $label: non-streaming run"
    local -a extra_plain=("$@")
    local -a plain_opts=()
    split_list plain_opts "$PLAIN_ARGS"
    if ! run_once "$build_dir" "$plain" "${plain_opts[@]}" "${extra_plain[@]}"; then
        record FAIL "$label non-streaming run"
        return 1
    fi

    log "-- $label: streaming run"
    local -a stream_opts=()
    split_list stream_opts "$STREAM_ARGS"
    if ! run_once "$build_dir" "$stream" "${stream_opts[@]}" "${extra_plain[@]}"; then
        record FAIL "$label streaming run"
        return 1
    fi

    if cmp -s "$plain" "$stream"; then
        local hash
        hash="$(md5_of "$plain")"
        record PASS "$label byte-for-byte streaming == non-streaming (md5 $hash)"
    else
        warn "output mismatch, see $plain and $stream"
        if command -v cmp >/dev/null 2>&1; then
            cmp "$plain" "$stream" >&2 || true
        fi
        record FAIL "$label byte-for-byte streaming == non-streaming"
        return 1
    fi
}

run_ctest() {
    local build_dir
    build_dir="$(abs_path "$BUILD_CPU")"
    if [ ! -d "$build_dir" ]; then
        record FAIL "ctest build dir missing: $build_dir"
        return 1
    fi
    if ctest --test-dir "$build_dir" --output-on-failure -R "$CTEST_REGEX"; then
        record PASS "ctest ($CTEST_REGEX) in $build_dir"
    else
        record FAIL "ctest ($CTEST_REGEX) in $build_dir"
        return 1
    fi
}

remote_target() {
    printf '%s@%s\n' "$REMOTE_USER" "$REMOTE_HOST"
}

ssh_run() {
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$@"
}

scp_run() {
    scp -o BatchMode=yes "$@"
}

run_remote_cuda() {
    local target script_quoted
    target="$(remote_target)"
    script_quoted="$(printf '%q' "$REMOTE_SCRIPT")"

    local repo_q model_q bin_q
    repo_q="$(printf '%q' "$REMOTE_DIR")"
    model_q="$(printf '%q' "$REMOTE_MODEL")"
    bin_q="$(printf '%q' "$REMOTE_BUILD_CUDA/bin/$BINARY")"

    log "-- preflight on $target"
    if ! ssh_run "$target" "cd $repo_q && test -f $model_q && test -x $bin_q"; then
        record FAIL "remote preflight (repo, model or binary missing on $target)"
        return 1
    fi

    log "-- copying $SCRIPT_NAME to $target:$REMOTE_SCRIPT"
    if ! ssh_run "$target" "mkdir -p $(printf '%q' "$(dirname "$REMOTE_SCRIPT")")"; then
        record FAIL "remote reachability ($target)"
        return 1
    fi
    if ! scp_run -q "$SCRIPT_PATH" "$target:$REMOTE_SCRIPT"; then
        record FAIL "copy script to $target:$REMOTE_SCRIPT"
        return 1
    fi

    local -a rargs=(remote-cuda
        --repo-root "$REMOTE_DIR"
        --model "$REMOTE_MODEL"
        --build-cuda "$REMOTE_BUILD_CUDA"
        --binary "$BINARY"
        --prompt "$PROMPT"
        --n-predict "$N_PREDICT"
        --seed "$SEED"
        --temp "$TEMP"
        --out-dir "$REMOTE_OUT_DIR")
    if [ -n "$STREAM_ARGS" ]; then rargs+=(--stream-args "$STREAM_ARGS"); fi
    if [ -n "$PLAIN_ARGS" ]; then rargs+=(--plain-args "$PLAIN_ARGS"); fi
    if [ -n "$CTX" ]; then rargs+=(--ctx "$CTX"); fi
    if [ -n "$UBATCH" ]; then rargs+=(--ubatch "$UBATCH"); fi

    local rargs_quoted
    rargs_quoted="$(printf '%q ' "${rargs[@]}")"
    local remote_cmd
    remote_cmd="flock -w $(printf '%q' "$GPU_LOCK_TIMEOUT") $(printf '%q' "$GPU_LOCK") bash $script_quoted $rargs_quoted"

    log "-- remote CUDA check on $target under flock $GPU_LOCK"
    if ssh_run "$target" "$remote_cmd"; then
        record PASS "CUDA byte-for-byte streaming == non-streaming on $target"
    else
        record FAIL "CUDA byte-for-byte streaming == non-streaming on $target"
        return 1
    fi
}

cmd_check() {
    if [ -z "$MODEL" ]; then
        die "check requires --model PATH (env MODEL)"
    fi

    if [ -z "$OUT_DIR" ]; then
        OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fork-sync-check.XXXXXX")"
    fi
    log "repo:  $REPO_ROOT"
    log "model: $MODEL"
    log "out:   $OUT_DIR"

    if [ "$SKIP_CPU" != "1" ]; then
        if run_pair cpu "$BUILD_CPU"; then :; fi
    else
        log "CPU pair skipped"
    fi

    if [ "$SKIP_CUDA" != "1" ]; then
        if [ "$REMOTE" = "1" ]; then
            if [ -z "$REMOTE_MODEL" ]; then
                record FAIL "remote CUDA model missing (--remote-model)"
            else
                if run_remote_cuda; then :; fi
            fi
        else
            if run_pair cuda "$BUILD_CUDA"; then :; fi
        fi
    else
        log "CUDA pair skipped"
    fi

    if [ "$SKIP_CTEST" != "1" ]; then
        if run_ctest; then :; fi
    else
        log "ctest skipped"
    fi
}

cmd_build() {
    resolve_repo_root
    command -v cmake >/dev/null 2>&1 || die "cmake not found"

    local cpu_dir cuda_dir
    cpu_dir="$(abs_path "$BUILD_CPU")"
    cuda_dir="$(abs_path "$BUILD_CUDA")"

    log "-- configuring CPU build: $cpu_dir"
    if ! cmake -S "$REPO_ROOT" -B "$cpu_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release; then
        record FAIL "configure CPU build"
        return 1
    fi
    log "-- building CPU (-j $JOBS)"
    if cmake --build "$cpu_dir" --config Release -j "$JOBS"; then
        record PASS "build CPU ($cpu_dir)"
    else
        record FAIL "build CPU ($cpu_dir)"
    fi

    log "-- configuring CUDA build: $cuda_dir"
    if ! cmake -S "$REPO_ROOT" -B "$cuda_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON; then
        record FAIL "configure CUDA build"
        return 1
    fi
    log "-- building CUDA (-j $JOBS)"
    if cmake --build "$cuda_dir" --config Release -j "$JOBS"; then
        record PASS "build CUDA ($cuda_dir)"
    else
        record FAIL "build CUDA ($cuda_dir)"
    fi
}

cmd_sync_prep() {
    resolve_repo_root
    command -v git >/dev/null 2>&1 || die "git not found"

    log "-- enabling rerere"
    git -C "$REPO_ROOT" config rerere.enabled true
    git -C "$REPO_ROOT" config rerere.autoupdate true

    local old_mirror new_mirror
    old_mirror="$(git -C "$REPO_ROOT" rev-parse --verify --quiet "$MIRROR_BRANCH" || true)"

    log "-- fetching $UPSTREAM_REMOTE master into $MIRROR_BRANCH"
    if ! GIT_TERMINAL_PROMPT=0 git -C "$REPO_ROOT" fetch "$UPSTREAM_REMOTE" "master:$MIRROR_BRANCH"; then
        record FAIL "fetch $UPSTREAM_REMOTE master:$MIRROR_BRANCH"
        return 1
    fi
    new_mirror="$(git -C "$REPO_ROOT" rev-parse "$MIRROR_BRANCH")"

    if [ -z "$old_mirror" ]; then
        log "old mirror: (absent)"
    else
        log "old mirror: $old_mirror"
    fi
    log "new mirror: $new_mirror"

    if [ "$old_mirror" = "$new_mirror" ]; then
        log "upstream did not move; nothing to sync"
        record PASS "sync-prep (upstream did not move, mirror $new_mirror)"
        return 0
    fi

    if [ -n "$old_mirror" ]; then
        local integrated
        integrated="$(git -C "$REPO_ROOT" rev-list --count "$old_mirror..$new_mirror")"
        log "upstream commits integrated: $integrated"
        if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$old_mirror" "$new_mirror"; then
            warn "mirror moved backwards (old is not an ancestor of new); human decision required"
            record FAIL "mirror ancestry ($old_mirror -> $new_mirror)"
            return 1
        fi
    fi

    if [ -z "$ARCHIVE_BRANCH" ]; then
        ARCHIVE_BRANCH="archive/$FORK_BRANCH-pre-sync-$(date +%Y%m%d)"
    fi

    log ""
    log "Next steps (run manually, this subcommand never pushes and never rewrites):"
    log "  git branch -f $ARCHIVE_BRANCH $FORK_BRANCH"
    log "  git switch $FORK_BRANCH"
    log "  git rebase $MIRROR_BRANCH"
    if [ -n "$old_mirror" ]; then
        log "  git range-diff $old_mirror..$ARCHIVE_BRANCH $MIRROR_BRANCH..$FORK_BRANCH"
    else
        log "  git range-diff <old-mirror>..$ARCHIVE_BRANCH $MIRROR_BRANCH..$FORK_BRANCH"
    fi
    record PASS "sync-prep (mirror $old_mirror -> $new_mirror)"
}

cmd_remote_cuda() {
    if [ -z "$MODEL" ]; then
        die "remote-cuda requires --model PATH"
    fi
    if [ -z "$OUT_DIR" ]; then
        OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fork-sync-check.XXXXXX")"
    fi
    run_pair cuda "$BUILD_CUDA"
}

print_summary() {
    log ""
    log "==== summary ===="
    if [ "${#RESULTS[@]}" -gt 0 ]; then
        local line
        for line in "${RESULTS[@]}"; do
            log "$line"
        done
    else
        log "no checks ran"
    fi
    log "================="
    if [ "$FAILURES" -gt 0 ]; then
        log "RESULT: FAIL ($FAILURES failed check(s))"
        return 1
    fi
    log "RESULT: PASS"
    return 0
}

main() {
    if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
        die "bash >= 4.4 required (namerefs and empty-array expansion)"
    fi

    if [ $# -gt 0 ]; then
        case "$1" in
            check|build|sync-prep|remote-cuda) SUBCOMMAND="$1"; shift ;;
            -h|--help) usage; exit 0 ;;
        esac
    fi

    parse_args "$@"
    resolve_repo_root

    case "$SUBCOMMAND" in
        check)       cmd_check || true ;;
        build)       cmd_build || true ;;
        sync-prep)   cmd_sync_prep || true ;;
        remote-cuda) cmd_remote_cuda || true ;;
        *)           die "unknown subcommand: $SUBCOMMAND" ;;
    esac

    if print_summary; then
        exit 0
    else
        exit 1
    fi
}

main "$@"
