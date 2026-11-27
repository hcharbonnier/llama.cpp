# Agent Workflow

Mandatory rule for every main session (main conversation) of this project.
It defines who does what, in what order, and how far execution is autonomous.

## 1. Purpose and Scope

Any task or action that produces persistent changes (code, tests, configuration, documentation, scripts, assets) is executed by a **worker**, never directly by the main session.

The main session is an **orchestrator**: it plans, delegates, verifies, requests fixes, requests merges, then relaunches.

Strict rules for the main conversation:

- It **never writes code**, never modifies working files, and **never runs tests**. All technical work happens in sub-agents: `task` for a one-off operation (merge, targeted verification, analysis), `agent_manager` for ongoing work.
- It is **primarily used to interact with the user**: obtaining guidance rules, asking questions, clarifying requirements, arbitrating checkpoints.
- It may verify, read-only, that everything that should have been committed was indeed committed after the agents' work.
- Its top priority is to **keep its context as small as possible**: no reading large files, no build/test logs, no full diffs, no raw results. That information stays in the sub-agents, which only report an actionable summary.

## 2. Roles and Channels

| Role | Channel | Responsibility |
| --- | --- | --- |
| Main session | Current conversation | Interacts with the user, breaks down and delegates work, drives review/fix/merge, checks commits, reads worker reports from their files. Does not write code, does not run tests |
| Worker | `agent_manager` tool (`worktree` mode) | Performs an independent unit of work on its dedicated branch and writes its report to a dedicated file |
| One-off sub-agent | `task` tool (`general` sub-agent) | Merge, targeted verification, analysis, or an operation that does not justify a persistent worker. Returns its final message normally, no report file |

A worker never merges itself. The main session never merges and never runs tests. The merge `task` sub-agent does not implement features.

Commit policy: a worker commits freely on its own dedicated branch, without any user approval. This does not extend to `master` or any shared branch, where the user green light rules of section 9 still apply.

## 3. Breakdown and Parallelism

1. Break the request down into **independent** units of work (distinct files/modules, no blocking dependency).
2. Launch independent units **in parallel** through a single `agent_manager` call containing multiple `tasks`.
3. Parallelism is **virtually unlimited**: no arbitrary cap (not 3, not any fixed number). The only limiting factors are the real number of independent units and machine capacity.
4. Dependent units are launched in a later wave, after the prerequisites have been merged.
5. One unit = one branch = one worker = one report file = one explicit scope. Never assign two unrelated units to the same worker.

## 4. Phase 1 - Delegation via Agent Manager

- Use `agent_manager` with `mode: "worktree"` to isolate each worker.
- Provide a short `name` and an explicit, readable `branchName` (e.g. `feat/auth-token`, `docs/api-reference`).
- The worker prompt must contain:
  - the objective and expected outcome;
  - the exact scope (allowed files/directories, what is out of scope);
  - project conventions to follow;
  - verification commands to run (lint, typecheck, tests);
  - the prohibition on merging, switching branches, or touching the main branch;
  - the authorisation to commit freely (including `Assisted-by: <assistant name>`) on its own dedicated branch, without any user approval;
  - the **absolute path of the report file** (section 5) and the instruction to write it;
  - the **autonomy rule** (section 4.1): never ask the user a question, write `STATUS: BLOCKED` in the report file instead.
- Do not specify a model unless the user explicitly asks: the worker inherits the current model.
- If the project is not a git repository, `worktree` mode is impossible: use `mode: "local"` and report it, or request repository initialization.

### 4.1 Workers Are Fully Autonomous (No User Question)

An Agent Manager worker **must never ask the user a question**:

- it never uses the `question` tool;
- it never waits for user input;
- it never stops to request a confirmation.

When it meets a decision, it **decides**, documents the decision in the report (section 5), and continues.

The only exception is a user checkpoint (section 9): the worker then writes `STATUS: BLOCKED` in its report file, with the question and the options, and stops. It never asks by chat or by tool.

This rule applies to Agent Manager sessions only. `task` sub-agents are in-process: they return their final message normally and may use `question` if the calling prompt allows it.

### 4.2 Worker Prompt Template

The main session reuses this template for every `agent_manager` task (adapt the bracketed parts):

```text
Unit: <unit-name>
Branch: <branchName>
Report file (absolute): <repo-root>/.kilo/reports/<unit-name>.md

Objective: <objective and expected outcome>
Scope: <allowed files/directories> / Out of scope: <excluded paths>
Conventions: follow the project rules, ASCII only, no emdash, no unicode.
Verification: <commands to run and expected result>.

Rules:
- Commit freely on your dedicated branch, with Assisted-by: <assistant name>.
- Never merge, never switch branch, never touch the main branch.
- You are fully autonomous: never ask the user a question, never use the
  question tool, never wait for user input. Decide, document the decision in
  the report, and continue.
- If a user checkpoint (section 9) is reached, write STATUS: BLOCKED in the
  report file with the question and the options, then stop.

Report (mandatory): as your LAST action, replace the WHOLE content of the
report file with the format of section 5, starting with STATUS: DONE,
STATUS: BLOCKED or STATUS: FAILED.
```

## 5. Report Contract (Report File)

The worker's report is **a file**, not its final chat message. The `agent_manager` prompt/reply path is unreliable: the worker often receives the instruction but cannot get its reply back to the main session. The file is the only source of truth.

### 5.1 Report File Path

- Directory: `<repo-root>/.kilo/reports/` (absolute path given in the instruction).
- File name: `<unit-name>.md`, where `<unit-name>` is the explicit unit name given in the instruction (same value as `branchName` or a short slug). Keep it readable, e.g. `.kilo/reports/docs-agent-manager-report-file.md`.
- The main session passes the **absolute path**. The worker writes to that exact file.
- The directory is local-only and ignored by git (section 13).

### 5.2 Report File Format

The first line is the status marker, with no leading blank line:

```text
STATUS: DONE
Branch: <exact branch name>
Unit: <unit-name>
Attempt: <1, 2, 3 ... incremented on each instruction>
Written: <UTC timestamp, YYYY-MM-DDTHH:MM:SSZ>
```

Then the report content:

1. **Work summary**: what was done, why, and which decisions were made (including autonomous decisions).
2. List of files created / modified / deleted.
3. Verification commands run and their results.
4. Known issues, risks, and unaddressed points.
5. Confirmation that no merge was performed.

Status values are exactly `DONE`, `BLOCKED` or `FAILED`, written as `STATUS: <value>`.

### 5.3 Write Rules

- The report is written as the **last action** of the worker, after commit and verifications.
- The whole content is replaced each time (write, not append).
- An incomplete report blocks the next step: the main session sends a new instruction to rewrite the file before any merge.

## 6. Waiting for a Report and Fallback

### 6.1 Preparation Before Prompting

Before sending an instruction, the main session prepares the report file:

1. It writes a report skeleton file with `STATUS: PENDING`, `Attempt: <n>` (`n` = previous attempt + 1, starting at 1) and the current timestamp. Writing a skeleton in the main session is allowed: it is coordination state, not project code.
2. It sets the `Attempt: <n>` value in the worker instruction ("write `STATUS: DONE` and `Attempt: <n>`").

The `STATUS: DONE` + `Attempt: <n>` pair identifies a fresh report: the previous content never matches, and a stale report cannot be mistaken for a new one. The main session never deletes a report file.

### 6.2 Waiting

The main session waits with **one** `background_process` (`action: "monitor"`) call, which blocks until the report is stable or times out:

```text
bash .kilo/scripts/wait-agent-report.sh --file <repo-root>/.kilo/reports/<unit-name>.md --quiet-seconds 10 --timeout 1800 --poll 2
```

The script waits until the file exists, then until its size and mtime stay unchanged for `--quiet-seconds` consecutive seconds, then prints the content on stdout and exits.

Script exit codes:

| Code | Meaning | Main session action |
| --- | --- | --- |
| 0 | Report stable, content printed | Read the report and resume |
| 2 | Timeout, content never became stable | Read the partial content if any, then fallback (section 6.4) |
| 3 | Timeout, file never appeared | Fallback (section 6.4) |

On a non-zero exit, the main session reads the file directly if it exists; the partial content is useful diagnostic, never a valid report.

Strict rules:

- **One blocking `background_process` monitor call.** No polling loop, no repeated `read` of the file, no `sleep` in the main session, no foreground blocking shell.
- The content is read **only once stable**. Never read the file while a worker is writing it.

### 6.3 Interpreting the Report

- `STATUS: DONE` with `Attempt: <expected>`: the report is valid, continue (section 7 or section 8).
- `STATUS: BLOCKED`: the worker hit a user checkpoint (section 9). Relay the question and options to the user, then send the answer as a new instruction.
- `STATUS: FAILED` or a missing/partial report: the unit is not delivered, treat as a blocker or relaunch a fix instruction.
- `STATUS: PENDING` or an old `Attempt`: the worker did not write the report, use the fallback.

### 6.4 Fallback

If the report never appears, stays unstable past the timeout, or carries the wrong attempt:

1. `agent_manager` (`action: "list"`) to inspect the worker state (busy, idle, offline, waiting, permission).
2. If the worker is idle without a valid report, `agent_manager` (`action: "prompt"`) to ask for a rewrite of the report file (same absolute path, same `Attempt: <n>`), then wait again (section 6.2).
3. If the worker asks a permission, resolve it in Agent Manager, then wait again.

**Never rely on the prompt reply content.** Any answer received through the tool is ignored: only the file counts. If the worker stays unreachable, the unit is declared blocked with an explicit report to the user.

### 6.5 Status Cycle

Every instruction opens a new status cycle: new `Attempt`, fresh skeleton, new wait. This applies to implementation (section 4), review/fix (section 7), and any follow-up. There is no valid report without the `Attempt` value expected for the current cycle.

## 7. Phase 2 - Review and Fix by the Worker

As soon as a worker report is `STATUS: DONE`:

1. Retrieve its `sessionID` via `agent_manager` (`action: "list"`).
2. Prepare the next report cycle (section 6.1) on the **same** report file: `Attempt: <n + 1>`.
3. Send instructions back to the **same** worker with `agent_manager` (`action: "prompt"`, `sessionID`, `prompt`), referencing the **same absolute report path**:
   - request a full review of its own diff (`/review worktree` in a managed worktree);
   - request fixes for **all** detected bugs and defects, including regressions, edge cases, type errors, and missing tests;
   - request that verifications be rerun after the fixes;
   - require a rewrite of the report file (section 5) with `Attempt: <n + 1>` when done.
4. Wait on the same file again (section 6.2) and check the new attempt.
5. Repeat the review as long as blocking findings remain. Each round is a new status cycle.

## 8. Phase 3 - Merge via `task` Sub-agent

Once review and fixes are complete, the main session launches a dedicated **`task` sub-agent** for the merge.

The `task` sub-agent is in-process: it keeps returning its final message normally, and the file-based report protocol does **not** apply to it.

The merger prompt must contain **at least**:

- the worker's **branch name**;
- the worker's **report content**, reproduced in full (read from the report file: summary + files + verifications);
- the target branch (current branch) and the repository/worktree path;
- instructions: merge without losing work, resolve conflicts while preserving both sides' intent, never force (`--force`), never rewrite history, never delete files outside the scope;
- the obligation to rerun lint / typecheck / tests on the merged result;
- the obligation to report: merged commit/branch, conflicts encountered and how they were resolved, verification results, and any failures.

Merge rules:

- Merges are **serialized**, one at a time, to limit conflicts.
- If verification fails after the merge, the merger does not validate: it reports back, and the main session sends the fix to the original worker or opens a new unit of work.
- For non-trivial conflicts (product or architectural decisions), escalate to the user (section 9).

## 9. Continuous Loop and Autonomy

As long as defined work remains for the project, the main session:

1. launches new workers via `agent_manager` as soon as workers have finished;
2. applies phases 2 and 3 to each worker;
3. updates the list of remaining tasks;
4. starts over without asking for intermediate confirmation.

The chain is **100% autonomous**: no "should I continue?" questions, no stopping after a single wave, no waiting for approval between review, fix, and merge.

The main session stops only if:
- all defined actions are complete and merged;
- or a user checkpoint (section 10) is reached;
- or a real blocker prevents progress, with an explicit report.

## 10. User Checkpoints (Mandatory Green Light)

Ask for approval **only** in these cases:

- destructive or irreversible action (data deletion, `reset --hard`, `push --force`, history rewriting, destructive migration);
- production impact, deployment, publication, external sending, spending;
- secrets, credentials, permissions, or account configuration;
- blocking ambiguity about requirements or a major product/architectural decision;
- non-trivial merge conflict involving a business decision;
- exceeding the originally requested scope.

Outside these cases, proceed without interruption.

A worker that reaches one of these cases does not ask: it writes `STATUS: BLOCKED` in its report file with the question and the options, then stops (sections 4.1 and 6.3).

## 11. Quality and Verification

- No worker delivers unverified work.
- Minimum verification: formatting/lint, typecheck, relevant unit tests, build when applicable.
- The merger re-verifies after merging.
- Never work around a failing test by deleting or disabling it; fix the cause or escalate.
- Never commit secrets.

## 12. Main Session Context Hygiene

The main conversation's context is a scarce resource: it must stay small and stable.

- Never run builds, tests, benchmarks, or log-generating commands in the main session.
- Never read full diffs, logs, large files, or test output in the main session. Delegate the inspection and keep only a summary.
- Exchanges with workers are limited to instructions, `sessionID`s, branch names, report paths, and concise report contents.
- Waiting for a worker is done through a single blocking `background_process` monitor call running `.kilo/scripts/wait-agent-report.sh`. The report content is read only once stable, and only once. No polling, no `sleep`, no foreground blocking shell.
- Main-session checks are limited to targeted reading and a git cleanliness check (`git status`, current branch) to confirm the expected work is committed.
- One unit of work per worker, one short report per unit: no catch-all.
- Exception: during a merge, the worker's report content is passed in full to the `task` sub-agent, but it is not re-analyzed in detail by the main session.

## 13. Local-Only Harness Files

These files are local tooling, never committed:

- `.kilo/scripts/wait-agent-report.sh`: the wait harness (section 6.2);
- `.kilo/reports/`: the report files (section 5.1).

They are listed in `.kilo/.gitignore`. Never `git add` them, never commit them.

## 14. Forbidden Anti-patterns

- Modifying files directly in the main session when the task is delegable.
- Writing code or running tests, builds, or benchmarks in the main session.
- Loading large logs, diffs, or test output into the main session.
- Relying on the `agent_manager` prompt reply for a worker result instead of the report file.
- Reading a report file before it is stable, or polling it in a loop.
- Using `sleep`, a foreground blocking shell, or a `read` loop instead of the wait script.
- Accepting a report whose `Attempt` is not the expected one.
- Deleting a report file instead of preparing a new skeleton.
- Letting an Agent Manager worker ask the user a question or wait for input.
- Artificially limiting parallelism to 3 workers.
- Merging without prior review and fixes.
- Merging without the branch name and the worker's latest report.
- Having the worker merge its own branch.
- Waiting for user approval when no section 10 checkpoint applies.
- Launching a new worker on a scope that overlaps an active worker without reason.
