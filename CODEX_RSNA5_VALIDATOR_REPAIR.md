# Codex: Repair RSNA 5% result validation on gpu04; preserve all 48 training runs

## Authorization and desired outcome

The user explicitly requests: **diagnose and repair the validator/provenance failure** marking 40 completed RSNA 5% schedule runs FAILED with uninformative `AssertionError()`; recover these runs as COMPLETE_VALID **only if independently verified**. Do not rerun a successful train, do not run another test inference, do not stop active jobs. Return exact cause(s), code/test changes, which runs validated/invalid/unresolved and numeric paired aggregate if valid groups complete.

This GitHub file is a Codex handoff only; its creation does not invoke Codex/SSH. Execute this in the user's active Codex session after fetching the branch.

## What is known (user's last report, not live status)

- Host: `ssh enzez@instgpu-04.cs.wisc.edu`, physical GPUs 0–7, all 8 were running **final seed-5 wave**, two backbones × four conditions, around epoch 15/30.
- Earlier **40 of 48** runs marked `FAILED` by a controller due to `AssertionError()`, despite *each* reportedly having: process exit code 0; complete 30-epoch history; finite final test MAE/R²; existing predictions; checkpoint reloaded; exactly one final test pass.
- Study: RSNA Pediatric Bone Age, 5% labeled, 30 epochs, backbones R50 and DINOv2-S/14, seeds 0–5, schedules FRONT, EVEN, BACK, DENSE_MASS_MATCHED, total 48 distinct keys. Per-epoch `U=len(unlabeled_loader)` updates; `L=len(labeled_loader)` anchors for FRONT/EVEN/BACK; DENSE at every update with coefficient L/U (nominal mass L). Unlabeled objective fixed weak-to-strong MSE with coefficient 1, sample weight 1. RSNA-specific original batch size 8, workers 2, drop_last=False, no gradient clipping, target months.
- The earlier task branch `tasks/rsna-anchoring-48run-20261117` contains `CODEX_RSNA_48_RUNS.md` with exact protocol. Read it if needed. Actual **as-run source/worktree on gpu04** and controller files are authoritative; do not assume public source snapshots are coherent or active.

## Safety: avoid corruption of live seed-5 wave

1. Inspect host identity, `nvidia-smi`, GPU UUIDs, process list and status, controller PID/tmux, real worktree, output root, controller scripts, queue manifest, `FAILED` records and logs, as-run trainer+validator hashes. **Read only initially.** Do not assume paths, PIDs or seed-5 state from stale report.
2. **NO GPU training/inference or GPU memory allocation for this repair.** Don't stop/preempt seed-5 workers or existing queue controller; no SIGINT or tmux send-keys, no live trainer or checkpoint changes, no deletions, no automatic retrying of the 40.
3. While any training/controller remains live, make any new validator implementation and tests in a **separate isolated audit folder/worktree** with entirely disjoint outputs, and run against **read-only copied snapshots or independent file reads** of the already-complete 40 jobs. Do **not** rewrite active `RUN_MATRIX`, `QUEUE_STATUS`, shared controller state, status files, or script it could still execute during active runs. Identify race hazards (controller might requeue FAILED entries): verify no retries are pending; if it would auto-retry, ensure no successful completed result is overwritten without interfering with the active jobs. Escalate if safe intervention isn't possible.
4. A validator may be fixed, but **training scientific behavior must never be changed**. Leave all original manifests, model checkpoints, histories, predictions and metrics byte-identical; hash them before/after relevant inspection. No test inference to reconstruct metrics, and no retroactive adjustment of hyperparameters, seed exclusions, or data labels.
5. Use least-invasive repair: if validation logic was wrong, correct in isolated validator code and add a regression test. Do NOT simply remove assertions, bypass checks, force `status=valid`, or change source hashes to match an incompatible run. Actual invalidities must remain invalid.

## Phase A: diagnose exact AssertionError

Locate concrete offending validator line/condition and full exception traceback: controller/subprocess stderr, validation helper, status transaction, terminal logs, manual CPU-only reproduce using existing completed run artifacts. Add a precise error message and log actual observed-vs-expected values. Compare multiple failed runs across seeds/backbones/schedules, and look for a systematic mismatch:

- Are the 40 flags due to wrong expected job key / schedule alias / seed/backbone names / path conventions, status enum (COMPLETE vs COMPLETE_VALID), an out-of-date config schema, a hard-coded count, or an incorrect drop_last or L/U assumption?
- Does a validator mistake a valid 30-epoch checkpoint reload/test-once status, wrong metric key names `test_mae_months` vs `test_mae`, or an epoch count?
- Does it compare native `metadata` SHA with an **after-the-fact validator/source hash** rather than the frozen as-run trainer hash? Does it use a wrong archived profile/snapshot? Do not change scientific provenance to make assertions pass.
- Could actual source/manifests/initial model digests, optimizer/scheduler/augmentation, unlabeled stream ordering, label mass/count, target scaler, or protected test protocol differ by schedule? A real mismatch invalidates that comparison.
- Validate path ownership and no duplicate attempt; trainer exit=0 and finite metrics are necessary but **not sufficient** for a scientific `COMPLETE_VALID` claim.

## Phase B: independent run-by-run validation (CPU / persisted files only)

For all 48 matrix cells, with expected status complete or running as actually observed:

- Exact dataset=RSNA, ratio 0.05, backbone, condition and seed; manifest/cohort/split hashes and original-target months; correct labeled-only scaler and seed mapping.
- As-run trainer SHA and config, optimizer, LR scheduler, 30 epoch budget, FP32, dataset-specific batch=8, 2 workers, no clipping, drop_last false; correct weak/strong augmentation roles and one `L_u` per optimizer step.
- Check actual logged `U` updates every epoch; FRONT/EVEN/BACK exact L anchors once each per epoch at stated indices, coefficient=1, no recycling; DENSE U anchors with coefficient L/U, nominal coefficient mass L; no extra optimizer steps; if available check unlabeled order/initial model hash paired across schedules.
- 30 complete history rows, checkpoint best epoch chosen by validation MAE only, saved best checkpoint metadata and reloaded confirmation; exactly one already-recorded held-out test pass; prediction count and original-order alignment; finite test MAE/R² (months) consistent with saved arrays. Recompute MAE/R² **from persisted prediction arrays on CPU only** if appropriate as an integrity crosscheck, not fresh inference; treat precision rounding correctly.
- Ensure all pairwise comparisons among 4 schedules share same manifest/seed/backbone and source/initialization protocol; explain any deviations.

Produce an **independent per-key audit** with statuses `VALID`, `INVALID`, `UNRESOLVED`, or `IN_PROGRESS`, reason codes, exact evidence paths/hashes and a traceback. No selective elimination or silent skipping of seeds.

## Phase C: targeted validator fix, safe reconciliation

1. Correct the **root cause** in an isolated local branch/file; write focused CPU-only unit/regression tests reproducing the original `AssertionError()` and proving the correction. Include tests for actual scientifically invalid artifacts so they remain rejected, not merely a happy path. Preserve old validator as immutable reference (hash).
2. After all 48 original jobs have naturally finished and the original controller has quiesced, re-evaluate the 40 old flags using the corrected validator. If and only if safe to update state and all scientific checks pass, make an **atomic, backed-up, idempotent** reconciliation to their existing run status records. If a live controller still owns status files, do not race it; preserve independent validation report only, reconcile later after safe handoff. Never mutate checkpoint, prediction or metric artifacts.
3. If original controller has truly ended and all valid runs pass: provide a **private** six-seed RSNA 5% table per backbone/condition with MAE and R² mean ± explicitly named SD convention; paired deltas FRONT–EVEN, BACK–EVEN, DENSE_MASS_MATCHED–EVEN, and exploratory paired t-intervals (n=6, df=5). If run status incomplete or one cell invalid, report only fully matched seed subsets and n; do not imply 48/48 valid.
4. If any cell remains genuinely invalid or provenance cannot be reconciled, clearly report it and proposed remedy, without launching retraining. Separate `training complete` from `scientifically validated`.

## Reporting and local deliverables

Store private reports on gpu04 in a new distinct report directory (not original raw artifacts): `RSNA5_VALIDATOR_ROOT_CAUSE.md`, `RSNA5_48_KEY_AUDIT.csv`, `RSNA5_VALIDATOR_TESTS.md`, `RSNA5_RECONCILIATION_LOG.md`, and `RSNA5_PAIRED_RESULTS.md` when justified. Show exact corrected code file path and before/after SHA, original failing traceback + exact assertion, the number of valid/invalid/unresolved/still running cells, seed-5 current PIDs/epochs, whether any controller/status file was modified and by what process, and how to repeat the CPU-only check. Do not upload unpublished numbers or protected medical data to public GitHub.

**High priority acceptance:** no successful training rerun, no repeated test inference, no live worker interrupted, no silent status overwrite, no fabricated results; Q3/Q4/other 24 GPU experiments unchanged. A GitHub handoff does NOT start SSH work by itself.
