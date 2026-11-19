# Codex task B — Q4 standard dense-loop audit + targeted clean dense TE on gpu02 GPUs 0–3

## Authorization / scope

The user approves using the **currently idle four gpu02 GPUs 0,1,2,3** to address the original-method / standard FixMatch-style mixed-supervision comparison, provided real-time checks still show them idle. The current 4–5 RSNA 10%/20% queue is NOT to be touched, paused, killed or requeued. GPUs 6–7 are reserved for the separate task C below; do not allocate them here. The user has already approved launching the **24 new clean-dense Temporal Ensembling runs** specified below after an implementation audit/preflight; do not expand to extra methods or seeds without an explicit written next task.

This document only instructs Codex; posting it on GitHub does not start any remote agent or job.

Host: ssh enzez@instgpu-02.cs.wisc.edu
Last user-reported snapshot (Nov 18, 2026 23:36 CST): gpu02 cards 0–3 and 6–7 idle; 4–5 busy with the RSNA extension. Verify live devices, UUIDs, PIDs, VRAM, worktrees, controller ownership and quota before using any card. Avoid disk/IO bottlenecks due to RSNA. No preemption, no oversubscription or modification of previous result directories.

## Part I: FIRST audit existing 5% Q4 matrix (read-only)

The requested comparison is each objective's **native** loop vs a true **standard dense mixed-supervision** loop, at four datasets × two backbones, using the existing six seeds where available. Inspect actual historical source-to-profile mapping in essential_code/profiles.json, local as-run configs/logs, paper result archives, existing +Interaction source, prior campaign outputs and metadata. Produce a scientifically precise matrix with cells existing/valid/missing or non-equivalent:

- Plain / Uniform: native early L anchors versus the already-existing k=1 dense +Interaction with one L_s at each U step (no repeat run if matching).
- Mean Teacher: native TwoStreamBatchSampler already gives labeled data each native step. Native MT is the standard dense counterpart **under its own native step convention**; MT+Interaction adds a SECOND independent L_s term and must not be mislabeled the same as native dense.
- Temporal Ensembling: native dataset-mixed iterator and masked labeled loss. Some mixed steps may have no labeled examples; TE+Interaction retains that masked loss and adds independent recycled L_s. Do not call either of these automatically the uniquely defined, clean FixMatch-style variant.
- HPL: audit official vs adapted schedule; HPL already has duplicated/recycled labels in the original method, and the +Interaction version changes epoch traversal (and potential labeled sample exposure). Distinguish original source vs paper adaptation and verify actual predictor-step count, labeled iterator behavior and meta updates before classifying it as standard dense.
- UCVME, SimRegMatch: adapted original loops are labeled-controlled; +Interaction may cycle labels for U steps. Inspect the *exact* run metadata, preserve method-specific pseudo-label/uncertainty objectives and check whether +Interaction constitutes an existing FixMatch-style dense loop without changing method hyperparameters.
- For every method list epoch-specific native U/L/actual supervised-containing optimizer steps (and any extra supervised branch), optimizer steps, labeled exposure, used original official repository/commit, and whether dense is literally the same experiment as an existing +Interaction result. Avoid invented numeric step counts.
- Explicitly flag SimRegMatch percentile code uses np.percentile(..., 0.95) rather than 95; this should be disclosed/audited separately, not silently repaired in a schedule comparison.

Prepare local PRIVATE report Q4_NATIVE_VS_DENSE_AUDIT.md and a compact table usable in rebuttal; no public reviewer names, raw experiment data or unpublished numeric results.

## Part II: execute 24 NEW, explicitly defined TE clean-dense runs

A real gap is **Temporal Ensembling's clean dense comparison**: we want one full coefficient-1 supervised MSE on a recycled labeled batch per mixed-step, **instead of** counting the native masked-supervised contribution AND an extra labeled loss as two supervised branches.

Use the formal per-dataset TE adapter's normal mixed iterator and temporal-ensemble target mechanism, same U_native=len(train_mixed_loader) optimizer steps, method-specific ramps, consistency loss, ensemble accumulation, snapshot update order and scheduler/optimizer. Define:

- Method: **Temporal Ensembling CLEAN dense**:
    L_step = [coefficient-1 MSE on one recycled labeled batch] + [native TE consistency term on original mixed train batch].
    Set the native batch's masked supervised component to **zero in the optimizer objective**; do not accidentally retain both. Retain the same mixed-batch forward pass and all TE temporal target storage and update semantics. Exactly one L_s term each optimizer step.
- Datasets: **UTKFace, IMDB-WIKI, RSNA Bone Age, CheXchoNet LVIDd** at **5% labels**.
- Backbones: **ResNet50**, **DINOv2 ViT-S/14**.
- Seeds: **0,1,2**.
- Epochs: **30**; **24 runs** (4 × 2 × 3 × 1 condition).
- No additional methods, seeds, hyperparameter tuning or label-ratio expansion under this task.
- Preserve each dataset's original formal TE protocol and any medical dataset-specific optimizer, batch size, normalization and augmentation conventions rather than blindly copying UTKFace defaults. Verify all methods are correctly adapted from the official s-laine/tempens trajectory/ensemble code. Document the fact that replacing the native masked loss changes the loss normalization relative to the native implementation; do not falsely claim identical objective coefficient mass.
- Preserve label split/target scaler from labeled train only, validation checkpoint selection, strict test isolation, and exact trainer/model source provenance. Prior native TE and TE+Interaction results may serve as *reference* but are not automatically strictly paired unless initialization/protocol/seed equality is proven. Report provenance and avoid exaggerated causal claims.

Implement fresh standalone worktree/outputs e.g. /nobackup/enzez/q4_clean_dense_te_20261118/artifacts/q4_clean_dense_te_20261118/, with 24 immutable job keys. Use only verified free gpu02 GPU 0–3, one process/card, detached restartable queue and live-PID checks. Save immutable per-run configs, dataset/manifest hash, initial state hash, actual supervised-containing step counts, TE temporal state checks, full 30-epoch history, validation-best reload, test-once MAE and R2, and run status. Do small FP32 real-data preflight on both backbones and all four datasets before large launch; any incompatibility becomes an explicit blocker, not a silent protocol change.

**Scheduling:** Part I audit first so existing runs are not pointlessly reproduced. Part II 24-run clean-dense TE is a defined NEW variant and should launch if preflight passes. Do not launch extra HPL/UCVME/SimRegMatch jobs merely because some existing artifacts are missing; list them for a later approved matrix. Do not run independently on GPU 4–5 or 6–7.

## Return to user

Give audit native/dense classification for all six methods; count valid existing cells; identify true gaps; report clean-dense TE 24-key queue statuses and verified gpu02 GPU 0–3/PIDs/tmux/root paths with monitoring commands. Report method adaptations and confounds transparently. Protect all RSNA, Campaign B/D, and task C jobs. All numerical/private results stay server-local, not pushed to public GitHub.
