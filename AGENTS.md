# ACTIVE Codex task — RSNA 5% validator repair on gpu04

Read [CODEX_RSNA5_VALIDATOR_REPAIR.md](./CODEX_RSNA5_VALIDATOR_REPAIR.md) **completely**, then execute its steps in the user's Codex session. This is the **only** active task on this branch. The old paper-only cleanup is not active; do not relaunch RSNA training or other previously running campaigns.

**Immediate user goal:** diagnose `AssertionError()` that marked 40 already-trained RSNA 5% mechanism runs FAILED despite trainer exit code 0, 30 epochs, finite metrics, and recorded test-once checkpoint reload. Audit provenance, fix a faulty validator only if this is truly a validator defect, and safely reconcile results. Preserve the remaining eight seed-5 workers (last reported around epoch 15/30) and their live queue controller on `instgpu-04.cs.wisc.edu`.

**Safety:** read-only examination first; no GPU tasks, reruns, test re-inference, stop/kill/preempt, overwriting checkpoints, altering scientific data/source, or changing shared controller/status files while a process owns them. Make tests/repaired validator in isolated folder; do safe backed-up idempotent status reconciliation only after the controller has naturally finished. Mark real scientific mismatches INVALID/UNRESOLVED, never force VALID by disabling a check.

Give the user exact traceback/assertion, independently validated counts, whether seed-5 is running, repaired validation logic and regression tests, private report paths, and results table with paired six-seed comparisons **only if** genuinely verified. This GitHub file alone does not start Codex or remote work.

**Privacy:** the repo is public; never commit draft manuscript, reviewer identities, medical data, unpublished results, local logs/checkpoints or credentials.
