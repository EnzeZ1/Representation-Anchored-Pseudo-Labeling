# Codex task C — Disentangle native supervision coefficient vs added labeled batches (gpu02 GPUs 6–7)

## Authorization / motivation

The user approves using the currently idle **gpu02 GPU 6 and GPU 7** to address a concrete interpretation issue in the original 5% Mean Teacher and Temporal Ensembling +Interaction comparisons. Their native training loops already receive supervised signal; adding an independent recycled labeled MSE changes *supervised loss mass/exposure*, not just the temporal placement of supervision. The 36 controlled runs below are explicitly bounded. If preflight fails, report blockers; do not silently alter the experimental matrix.

Posting this task on GitHub does NOT invoke Codex or launch jobs. The user must issue a command in their active Codex coding session.

Host: ssh enzez@instgpu-02.cs.wisc.edu. Last user snapshot Nov 18 2026 23:36 CST shows GPUs 6,7 idle; **4,5 reserved for ongoing RSNA**, 0–3 reserved for independent Q4 clean dense TE task. Inspect live UUIDs/PIDs/VRAM, controller queues, disk/quota and memory before launch; no interference, one worker per genuinely available card. Do not use gpu01/03/04, and do not edit any earlier run artifacts.

## Exact 36-run matrix (UTKFace 5%)

- Dataset: **UTKFace**, canonical 5%-labeled manifests.
- Methods: **Mean Teacher (MT)** and **Temporal Ensembling (TE)**.
- Backbones: **ResNet-50** and **DINOv2 ViT-S/14**.
- Seeds: **0,1,2**, 30 epochs.
- Three conditions, all initialized fresh and identically for matched method/backbone/seed:
  1. native: keep the original native optimizer objective, with its existing labeled supervision L_s_native and its method-specific consistency L_u.
  2. doubled_native_supervision: use 2 * L_s_native + L_u (unchanged native stream and optimizer-step count; no new labeled batch). Do not apply extra optimizer/EMA update.
  3. extra_recycled_supervision: use L_s_native + L_s_extra + L_u, with **one fresh batch from a recycled labeled-only loader** per *same native optimizer step*, no second optimizer/EMA update.
- Count: 2 methods × 2 backbones × 3 seeds × 3 conditions = **36**.
- This is **not** a 1:1 equivalence test in TE: its native masked loss may be zero when a mixed batch contains no labels, while the recycled extra term is full labeled-batch MSE every step; doubling native supervised coefficient is therefore not guaranteed to match actual supervised loss mass. Record observed native labeled batch counts, normalization factors, per-term losses and labeled sample exposure, and qualify comparisons accordingly.
- MT native two-stream sampler has labeled examples every native step; the native/mass-increase/extra-batch triplet is a closer test of the question whether the improvement stems from additional labeled examples or simply a larger coefficient on an existing supervised term.
- For TE, preserve original temporal target buffers, ensemble correction, ramp schedules and mixed-data forward passes; for MT, preserve EMA student/teacher updates, two-stream sampler, and consistency weighting. No alteration of pseudo-label construction, extra separate optimizer steps, auxiliary uncertainty, or hidden hyperparameter changes. All methods use 30 epochs, the as-run dataset-specific UTKFace protocol, proper validation-best selection and exact 5% manifests.
- Use original as-run baseline recipes and official adaptation references. Prior historical MT and TE +Interaction source may already exist, but run these three conditions in **one coherent dedicated trainer** to ensure strictly paired initial hashes/protocols and avoid confusing different seed/initialization/RNG campaigns. Keep all source snapshots in essential_code immutable; use essential_code/profiles.json to identify coherent baseline code.

## Implementation and safe execution

1. Check SSH, GPU identity and occupancy 6–7; ensure existing RSNA 4–5 controller and Q4 0–3 queue are untouched. Distinct worktree/results root e.g. /nobackup/enzez/mt_te_weight_ablation_20261118/artifacts/mt_te_weight_ablation_20261118/.
2. Explicit 36 run keys, check correct per-seed 5% cohort, no leakage, labeled-only target scaler, model-state initialization equality, optimizer/scheduler/bootstrap, dropout/ramp, native step counts, and role-isolated RNG when possible. Do not assume identical raw native batch sequence when adding new recycled loader if RNG is not isolated; detect and document.
3. CPU/synthetic tests confirming per-method native steps, one optimizer step (and one EMA step for MT) per iteration, exact objective component and coefficient, no unlabeled loss definition change; check independent labeled stream.
4. GPU FP32 smoke tests on gpu02 6/7 with BOTH backbones using the original as-run batch sizes, no silent AMP/precision/image-size/batch/optimizer changes on OOM. If requested conditions are infeasible, mark them blocked and explain.
5. Launch detached, restartable queues on verified available gpu02 6 and 7, one worker per card, with controller lock and distinct run names. Verify live worker/controller PIDs and advancing finite losses, not only enqueued manifest.
6. Per valid run: source, protocol and manifest hashes, same initial parameter SHA256, actual native/recycled label exposure, per-term loss curves, best-val checkpoint history and reload, test exactly once after validation selection, MAE/R2 years, seed/pairwise aggregate with n and uncertainty. Keep all results, configs and image paths on private cluster only.
7. Save private PREFLIGHT_REPORT.md, RUN_MATRIX.csv, QUEUE_STATUS.md, METHOD_OBJECTIVE_AUDIT.md, aggregate_results.py and monitoring instructions. Analyze within paired seed differences: boosted-native vs native, extra-recycled vs native, extra-recycled vs boosted-native; no unsupported assumption that coefficient mass equals gradient mass.

## Result interpretation

The purpose is to distinguish (a) increasing a native supervised coefficient from (b) adding new sampled labeled batches to an already-supervised SSL method. For Mean Teacher this directly targets the concern that the published +Interaction is merely a stronger L_s; for TE, note the native masked-supervision peculiarity and do NOT conflate double-weight vs full recycled-batch exposure. Do not call these tests proof of novel label recycling or universal superiority.

Return 36-run completed/running/queued/failed count, exactly which of gpu02 6/7 are used, live PIDs/controller and log paths, any source/config deviations, and interim valid results only when truly complete.
