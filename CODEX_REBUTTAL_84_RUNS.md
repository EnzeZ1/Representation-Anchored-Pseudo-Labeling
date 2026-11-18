# Codex execution task: 84 additional rebuttal experiments on gpu03/gpu04

## User authorization / goal

The user explicitly approved two extra focused campaigns. **Implement and launch the 84 authorized experiments on idle GPUs of gpu03 and gpu04**, after verifying actual machine availability and scientific compatibility. Do not ask for further permission for those 84 experiments when preflight succeeds. Do not expand the number of seeds, methods, backbones, or data ratios.

The task is NOT a paper release and NOT a rerun of earlier completed experiments. It is separate from the **192-run 2080 Ti campaign** (gpu01 Campaign A / gpu02 Campaign B) and the **original 48-run RSNA 5% campaign** previously launched on gpu04. Protect all of those jobs, results, and controllers. No preemption, kills, GPU oversubscription, cleanup of dirty checkouts, or overwrite of scientific files.

**Remote entry points** (confirm DNS/SSH before any action; earlier machines used the longer .cs.wisc.edu suffix):
```bash
ssh enzez@instgpu-03.cs.wisc.edu
ssh enzez@instgpu-04.cs.wisc.edu
```
Verify with `hostname`, `nvidia-smi`, process list, filesystem mounts, and controller logs. Previous observation of gpu03 "occupied" and gpu04 "running RSNA" may be stale; the user's latest observation is that machines 3/4 are idle. **Live checks decide.** These machines may each have 8 GPUs but never assume a card is actually free. Use **up to 8 genuinely idle and VRAM-suitable GPUs per host**, with 1 training process per GPU and zero interference with third-party processes. If idle GPU counts are lower, use the safe subset; if an existing controller or RSNA trainer remains alive on gpu04, leave it untouched and do not repurpose its GPU. A completed controller may have orphan GPU processes; check UUID/VRAM, not only `tmux ls`.

The user wants **execution**, not just a plan. If validation passes, launch the queue on the available cards, detach safely, and confirm live trainer PIDs and GPU UUID affinity. If blocked, report exact blockers; do not claim launch.

## Campaign C — RSNA Bone Age label ratio extension: 48 runs

- **Host preference**: gpu03.
- Dataset: original canonical **RSNA Pediatric Bone Age** benchmark.
- Labeled ratios **10% and 20%** (0.10, 0.20).
- Backbones: **ResNet-50**, **DINOv2 ViT-S/14** (full fine-tuning).
- Conditions (4): **front, even, back, dense_mass_matched**.
- Seeds: **0,1,2** (3 paired seeds only).
- Epochs: **30**.
- **Total: 2 ratios × 2 backbones × 4 conditions × 3 seeds = 48 runs.**

**Scientific objective:** exactly the RSNA Plain/Uniform weak-to-strong consistency objective, `L_u = MSE(f_theta(strong(x_u)), stopgrad(f_theta(weak(x_u))))`, `lambda_u = 1`, pseudo-label sample weight `r = 1`. Labeled batch target uses the seed/ratio-specific **labeled-set scaler**, MSE. Keep all RSNA-specific split, anatomy-preserving augmentation, evaluation unit (**bone age months**), pretrained weights, optimizer, checkpoint selection and data-loader conventions. No teacher/uncertainty/probe terms.

**Use the previously started RSNA 5% 48-run implementation as the primary reference**, located on gpu04 in the earlier experiment worktree/results if accessible. Reuse its exact runner/trainer/config *only after checking source hashes and that ratio generalization is scientifically valid*. Historical code starting points: `essential_code/shared/training/rsna_bone_age_uniform_formal.py`, `essential_code/shared/training/rsna_bone_age_plain_original.py`, `essential_code/shared/data_processing/rsna_bone_age.py`, plus `essential_code/profiles.json` for historical profile mapping. Do not treat `essential_code/shared` as a coherent runnable profile by default. The earlier 5% campaign is a separate result; do not modify or rerun it.

**Verified RSNA historical settings to cross-check**: 30 epochs, batch size **8**, 2 DataLoader workers, **drop_last=False** in the recorded RSNA Uniform loader, end-to-end fine-tuning. For ResNet-50 use Adam encoder lr 1e-4/head lr 1e-3, weight decay zero, StepLR(step 10, gamma 0.1). For DINOv2 use AdamW backbone lr 1e-5/head lr 1e-4, weight decay 0.05, CosineAnnealingLR(T_max=30). The historical RSNA Uniform trainer used **no gradient clipping**; do not silently port UTKFace's clipping norm 5 or batch size 32. If the earlier 5% experiment intentionally deviated, identify and reconcile the difference explicitly before running this extension; document comparability to 5%.

For each seed+ratio compute `U=len(unlabeled_loader)` and `L=len(labeled_loader)` dynamically; assert `0 < L <= U`. All four conditions use **exactly U optimizer updates per epoch and a complete unlabeled-loader traversal**, with the identical unlabeled loss. They differ only in supervised schedule/coefficient:
- `front`: `L_s` at indices `0,...,L-1`, strength 1, one labeled pass/epoch.
- `even`: `L_s` at `floor((j+0.5)*U/L)`, j=0..L-1, strength 1, one labeled pass/epoch.
- `back`: `L_s` at `U-L,...,U-1`, strength 1, **pure** back cluster (no early anchor), one labeled pass/epoch.
- `dense_mass_matched`: at all U steps, `(L/U) * L_s`, cycling labeled samples as required, with **nominal total supervised coefficient mass L**.

**Interpretation caveat:** front/even/back isolate supervision *placement* at fixed labeled exposure and supervised-update count. Dense mass-matched controls total nominal coefficient mass, **not actual gradient mass or labeled-example exposure**; clearly separate these claims.

Seeds must use official existing ratio=0.10/0.20 manifests and the canonical cohort. Validate no split leakage, manifests/target scalers, per-seed L/U, and data/weights access. If manifests need deterministic generation, use only the exact existing RSNA protocol without changing official split and do not overwrite manifests; record provenance. All four conditions for any seed+ratio+backbone should have matching initial model state hashes, model code, optimizer, scheduler, augmentation, training/validation/test splits and unlabeled sample order. Ensure labeled recycling does not perturb unlabeled RNG.

## Campaign D — UTKFace 1%/10%/20% supervised-only controls: 36 runs

- **Host preference**: gpu04 **only after verifying its previous RSNA 5% campaign is finished and GPUs are actually free**.
- Dataset: canonical **UTKFace**.
- Ratios: **0.01, 0.10, 0.20**.
- Backbones: **ResNet-50** and **DINOv2 ViT-S/14**.
- Modes (2): `sup_only_one_pass`, `sup_only_recycled`.
- Seeds: **0,1,2** only.
- Epochs: **30**.
- **Total: 3 ratios × 2 backbones × 2 conditions × 3 seeds = 36 runs.**

Both modes use labeled data ONLY: `L_s = MSE(f_theta(x_l),y_l)` on standardized labels, **no pseudo-label or unlabeled consistency term**.
- `sup_only_one_pass`: exactly `L` supervised optimizer updates per epoch, one labeled loader traversal.
- `sup_only_recycled`: exactly `U` supervised optimizer updates per epoch, cycle labeled batches; `U` must match the corresponding ratio/seed's full unlabeled trajectory length in **gpu01's currently running 144-run UTKFace ratio/frequency/placement campaign**.
Do not falsely claim these two modes have matched optimizer counts or exposure. `sup_only_recycled` is matched in supervised exposure/optimizer updates to the `k=1` dense Plain/Uniform training in the other campaign, provided data/optimizer/seed/initialization matches.

**CRITICAL for comparability:** The preceding 144-run UTKFace campaign (gpu01) may have generated new ratio=1/10/20% manifests and a dedicated ratio-generalized trainer. Before preparing this campaign, **read the actual 144-run campaign's frozen config/manifests and source revision**, and reuse the exact canonical seed/ratio memberships, image transforms, label scalers, `drop_last` semantics, optimizer/scheduler/checkpoint selection, model initialization and batch routing. If gpu04 cannot access gpu01's exact manifests, transfer only suitable non-sensitive metadata/manifests through authorized shared storage or other verified local workflows; never recreate an inconsistent split or silently pair nonidentical datasets. Do not touch gpu01's live jobs.

Historical starting points: `essential_code/shared/supervision_quantity_control.py` (legacy UTKFace 5%-only quantity control), `essential_code/shared/training/supervised_benchmark.py`, `data_processing/utkface_protocol.py`, previous 144-run campaign's implementation. Legacy modules hard-code 5%: **generalize explicitly, not a blind direct execution**.

Maintain the as-run UTKFace baseline: batch size **32**, 30 epochs, ImageNet normalization, weak labeled augmentations, full fine-tuning, correct dropout, ResNet50 Adam lr 1e-4 weight decay 1e-3 / StepLR(10,0.1), DINOv2-S AdamW backbone lr 1e-5, head lr 1e-4, weight decay 0.05 / cosine(T_max=30), **grad clipping norm 5**, and validation-best checkpoint. Inspect the frozen 144-run trainer to confirm its exact values; if mismatch found, report it, do not silently retune.

Validation/test data are **evaluation-only**; choose checkpoint using validation MAE (years) and only evaluate held-out test **once after reloading best checkpoint**. Record exposure count, exact optimizer-step count, initial state hashes, cohort/manifest hash and source hash per run.

## Cluster preflight, GPU safety and deployment

Before running ANY new scientific jobs:
1. Verify DNS/SSH, machine identity, physical GPUs, GPU UUIDs, currently running PIDs, memory availability, `tmux` sessions, any previous RSNA controller still alive on gpu04, disk free and user quota. A GPU is not free just because usage is low at one instant. Do not kill or detach other jobs.
2. Confirm actual source/worktree path, version/commit, dependency environment, pretrained weights, RSNA/UTKFace image datasets, protected-data access where relevant, and manifests. These GPU machines may not share the same paths despite familiar hostnames; test this.
3. Build **two separate, deterministic run manifests** containing exactly 48 and 36 distinct jobs, each with unique host/GPU assignment and output path. Never claim all 16 GPUs are available without observation.
4. Implement CPU/synthetic schedule-contract assertions (U updates for all RSNA conditions, L anchors in front/even/back, full 1 pass there; U anchors with coefficient L/U in dense, nominal mass L; L vs U supervised updates for UTKFace controls). Assert all four RSNA conditions share the unlabeled loss and optimizer setup.
5. Run a short **GPU preflight for BOTH backbones at original per-dataset batch sizes and FP32** on available devices before launching large training. Check finiteness, VRAM headroom, checkpoint/reload/validation path; **do not silently reduce batch size, precision, image size, trainable parameters, augmentation, or scheduler** on OOM. Report blocked cells instead.
6. Use a **separate git worktree / new scientific script files**, not the dirty source checkout; preserve historical code, results, branches and all artifacts. Avoid publishing under-review/private data or intermediate numerical results on public GitHub.
7. Start separate, **restartable detached queues** on the respective host(s), using a file lock/single controller per queue, one process per idle GPU, capped workers and disk/IO use, fresh result roots such as `artifacts/rebuttal_extensions_20261118/{rsna_ratios,utkface_sup_only}`. If host paths are not shared, keep disjoint host-local outputs and a global index; don't assume cross-host shared locks.
8. Once launched, verify the first wave's live trainer PIDs and GPU UUID affinity, record per-host controller PID/tmux session and logs. Do not claim 84 have finished when 84 have only been enqueued. Do not start a new job on an occupied GPU. Do not touch the ongoing 192-run tasks on gpu01/02.

## Reporting / required artifacts

For every run, persist config, source SHA/commit, baseline reference, manifest digest, ratio/seed, dataset/backbone/condition, GPU UUID/host/PID, software environment, `U,L`, optimizer count, nominal supervision coefficient mass, supervised update count, labeled exposure, 30-epoch train/validation history, best-validation checkpoint, its reload verification, exactly one final test evaluation, finite test MAE & R2 in original units and ordered test predictions stored **locally only**. No test-based model selection or opportunistic seed skipping.

Produce on the private cluster:
- `PREFLIGHT_REPORT.md` — actual gpu03/gpu04 free GPU inventory and scientific/protocol checks, any OOM/missing data/old RSNA activity;
- `RUN_MATRIX.csv` or JSON — all 84 unique planned keys and actual state;
- `QUEUE_STATUS.md` — timestamp, controllers/worker PIDs and GPU mapping, completed/running/pending/failed/blocked by campaign, plus safe monitoring commands;
- `aggregate_results.py` — seed-matched comparisons + per-ratio/backbone MAE/R2 mean/SD and 95% CI on paired differences when enough seeds actually finish, with n and SD convention explicit;
- `README_EXECUTION.md` — how to monitor and resume only if necessary (do not auto-restart failed jobs without verifying their cause).

**Response to user:** give exact SSH hostnames, verified free GPU IDs, counts launched/completed/pending for each of the 48/36 campaigns, controller PID + at least one verified live trainer PID, log/monitoring commands, deviations, and whether older RSNA 5% and existing 192-run campaigns were left intact. If resources are still busy, explain the real blocker instead of claiming launch. Keep this task focused on these 84 runs; no public release, no new seed expansion, no all-method experiments.

**Important:** This file is Codex's instructions. Uploading it to GitHub does not itself invoke Codex or start SSH jobs; the user will invoke Codex in its coding session.
