# Codex: SAFE RECOVERY of interrupted HPL factorial IMDB-WIKI runs (after host reboot)

## Language and purpose

**English ONLY** in Codex CLI status, progress, explanations, reports and final reply. The user communicates in Chinese in ChatGPT but requires all Codex/GitHub work in English.

The user previously authorized the 48-run HPL supervision-schedule × mechanism experiment under `tasks/hpl-factorial-schedule-vs-bilevel-20261120/CODEX_HPL_FACTORIAL_SCHEDULE_VS_BILEVEL.md`. This task **continues only the already-authorized 48-run core** after unexpected overnight host reboots. No new factorial conditions, ratios, seeds, or no-BLO P1 expansion are authorized here.

**Operational goal:** recover the 24 IMDB-WIKI 5%-label jobs from the prior validated original manifest, without losing or corrupting data or duplicating completed cells, and with scientifically correct matched training protocols.

## Evidence at Nov 21 2026, 10:28 America/Chicago (VERIFY again; do not presume current status)

- `gpu04`: UTKFace: **24/24 completed, independently validated**; do NOT rerun, modify, or test these artifacts.
- `gpu01`: IMDB: 0 complete, 8 interrupted, 8 previously queued. Host reboot reported at 03:19 CST. Eight seed-0 jobs stopped after 3–6 completed epochs.
- `gpu02`: IMDB: 0 complete, 4 interrupted, 4 previously queued. Host reboot reported at 02:51 CST. Four dense seed-2 jobs stopped after 1–2 completed epochs.
- At snapshot, controllers/tmux sessions were absent, no training active. **12 interrupted + 12 queued** IMDB cells (24 total). All three hosts were reported idle at snapshot, but this may have changed.

Consult prior original task instruction on the original branch as well as original private local code/worktree, `PRECHECK_AND_PROTOCOL.md`, `RUN_MANIFEST.csv`, `STATUS.md`, job manifests and controller logs. Do not reconstruct experimental cell definitions from memory or paper metrics. Preserve dataset/backbone/seed/schedule/mechanism/source SHA, U/M, num_epochs, validation checkpoint selection, one test evaluation, and meta sampling design.

## Phase 1: audit live state before touching jobs

1. Read original manifest and per-run outputs on the relevant hosts. Cross-check all **48 cell IDs**, and classify each one from actual artifacts: VALID_COMPLETE, PARTIAL_CHECKPOINT, PARTIAL_NO_VALID_RESUME_STATE, NOT_STARTED, LIVE, or FAILED. Check whether another agent restarted controllers after 10:28; if running, **do not duplicate**. Verify 24 UTK are complete and immutable.
2. Verify current host uptime, service state, GPU free memory/utilization, quotas, storage, and active jobs. Never assume all GPUs are still idle and never preempt other users.
3. Independently inspect partial checkpoint integrity and whether checkpoint contains complete **model, uncertainty learner, optimizers, LR schedulers, actual epoch, RNG (Python/NumPy/Torch/CUDA), data-sampler order/state, training-specific meta/unlabeled/labeled streams, and any relevant AMP state**. Check emitted test_count / validation history. Report whether bitwise-equivalent scientifically valid resume is possible for each interrupted cell.
4. Check saved controller PIDs and job IDs against processes actually alive. Do not treat stale RUNNING status or abandoned tmux sessions as live.

## Phase 2: safe recovery strategy

- Prefer faithful **resume** only if the exact required state is restorable and training continuation preserves pre-registered protocol, optimizer/scheduler trajectory, data stream, and one-time test guard. Never resume from a weights-only best-validation checkpoint as if it were a full optimizer/RNG state.
- If exact state is **not** available, do **not** manufacture continuity. Restart affected interrupted cells **from epoch 0** with the SAME seed, identical original source and protocol and a clean new attempt directory, retaining historical partial artifacts unchanged. Preserve original attempt directories as read-only; do not overwrite checkpoints or run histories. Validate that deterministic reinitialization and data split match original manifest.
- For 12 previously queued cells, launch as originally planned after reconciling whether they are truly not started. Do not rerun VALID_COMPLETE cells.
- If a restarted cell had a stale RUNNING marker, record the interrupted legacy attempt in a **separate recovery mapping/status index**; do not silently rewrite old artifact history. Make it easy to trace old attempt → new attempt.
- Use only approved idle GPUs and a robust controller that survives terminal disconnection; configure persistence across reboots where authorized without changing shared system services (e.g. controller logs/manifest-based recovery). **Do not require or attempt privileged service changes.**
- Guard against another reboot: atomic status and last-epoch tracking; periodic full-state checkpoints only for this new recovery if compatible with experimental protocol; avoid modifying actual loss/optimizer/scheduling behavior. A full-state periodic checkpoint is preferred if it can be added without changing the as-run numerical protocol. Verify controller restart/recovery is idempotent and won't trigger double tests.
- **No duplicate simultaneous attempts** for any cell. Lock per-cell at launch and use stable experiment IDs. Preserve all seed/config/source binding hashes.
- Give priority to finishing the originally authorized **24 IMDB core runs**. Do not start P1/no-BLO 24 extra runs or any new experiment design.

## Phase 3: validation and reporting

- Check every completed IMDB artifact: coherent `history.csv` for 30 epochs, checkpoint and metadata, best validation selection, finite metrics, expected U/L/schedule and meta-update counts, source/config hashes, final test evaluated ONCE, saved predictions where expected, and target split correctness.
- Independently verify paired seed mapping: two backbones × seeds 0,1,2 × 2 schedules (sparse/dense) × 2 mechanisms (Uniform/Full HPL).
- Report exact total `UTK 24 valid + IMDB completed = X/48`, in-progress/queued/interrupted/failed, hosts/GPUs, number restarted-from-zero vs truly resumed, and full local artifact paths in a **private** `HPL_REBOOT_RECOVERY_STATUS.md` and `HPL_REBOOT_RECOVERY_ATTEMPTS.csv` in the original private experiment artifact area or a new private recovery subdirectory.
- When IMDB completes, aggregate 48-run main effect results using matched seeds and population SD (ddof=0) with **paired differences** and cautiously interpreted exploratory confidence intervals. The effect of Full HPL vs Uniform **bundles** uncertainty weighting and bilevel plus extra auxiliary labeled meta feedback; do not call it an outer-only effect.
- Results may vary across R50 and DINO. Do not frame as “HPL gains are primarily from supervised density” unless the actual factorial supports that claim in the target setting.
- Keep test-only-once guard even after a reboot. If a completed cell has already tested, never run test again.

## Boundaries

- User authorization is for **recovery/completion of the previously approved 48 core runs only**. This is not permission to launch the conditional 24 P1 runs, a new 6-seed expansion, or any HPL provenance repairs.
- Never disturb UTK completed runs, ongoing unrelated campaigns, other users' processes, shared GPUs or service machines.
- Do not alter any old partially written experiment files, overwrite failed attempts, change historical code, change original outputs, edit manuscript/rebuttal, or push results to public GitHub.
- Source file changes for recovery robustness, if absolutely required, must live **only in an isolated recovery worktree/version and be recorded**, with a check that mathematical training semantics are unchanged.
- If cluster access is blocked or safe resumption cannot be done, return precise blocker and proposed manual next action; do not claim success prematurely.

## Final answer in Codex CLI (ENGLISH)

Concise status with: new audit snapshot time, what was verified, count of already-valid vs interrupted vs queued, exact number restarted/resumed/started, host/GPU allocation, run IDs and private paths, one-time test integrity checks, and whether recovery controller is actually active. Explicitly state if no jobs were launched. No fabricated ETAs.
