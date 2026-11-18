# Codex execution task: RSNA fixed-budget supervision study (48 GPU runs)

## User authorization and goal
The user requests that Codex **implement and launch this experiment now on gpu04**, using **up to 8 idle physical GPUs concurrently**. The agreed study is **RSNA Pediatric Bone Age, 5% labeled data, Plain/Uniform weak-to-strong consistency, two backbones, four supervision schedules, six paired seeds**: 2 × 4 × 6 = **48 runs**. This is a new controlled mechanism experiment. Do not expand to all six semi-supervised regression methods, additional datasets, or other label ratios.

This GitHub file is a handoff, **not evidence that the Codex agent or any GPU job has started**. When invoked by the user, perform the work in the gpu04 environment and report observed job IDs/PIDs; do not report success before verifying live processes. Do not ask for confirmation for this already approved 48-run experiment unless resource access or a substantive scientific mismatch is blocking execution.

## Source of truth and existing code
- Repository: `EnzeZ1/Representation-Anchored-Pseudo-Labeling`.
- Use the scientifically preserved historical source collection in `essential_code/`, especially `essential_code/profiles.json` and `essential_code/SOURCE_INDEX.csv`. It is **not** valid to treat `essential_code/shared/` as a guaranteed combined executable version. Establish a coherent source/profile or create a dedicated new experiment implementation without altering historical files.
- RSNA formal Uniform reference: `essential_code/shared/training/rsna_bone_age_uniform_formal.py`.
- RSNA Original Plain reference: `essential_code/shared/training/rsna_bone_age_plain_original.py`.
- RSNA anatomy-preserving transforms / split and manifest validation: `essential_code/shared/data_processing/rsna_bone_age.py`.
- UTKFace mechanism references, for **schedule ideas only** (not for copying its optimizer/dataset assumptions): `essential_code/shared/training/supervised_spacing_decoupling.py`, `essential_code/shared/training/supervised_mass_matched.py`, `essential_code/shared/training/canonical_uniform_frequency.py`.
- If local gpu04 paths, versions, or configs disagree with those references, stop on the affected discrepancy, report exact file hashes, and verify the as-run protocol before proceeding. Do not silently choose a different scientific configuration.

**Important RSNA-specific details verified in the preserved reference:** `rsna_bone_age_uniform_formal.py` defaults to 30 epochs, batch size 8, and 2 DataLoader workers; all of the labeled/unlabeled loaders use the RSNA manifest and have `drop_last=False` (no drop_last argument). Validation checkpoint selection is lowest validation MAE in **months**, test inference is only after loading the best checkpoint. ResNet50 uses Adam(encoder LR 1e-4, head LR 1e-3, weight decay 0), StepLR(10, gamma 0.1); DINOv2 ViT-S/14 uses AdamW(backbone LR 1e-5, head LR 1e-4, weight decay 0.05), CosineAnnealingLR(epochs); the reference has **no gradient clipping**. Maintain the RSNA transforms, optimizer, pretraining, target normalization from the labeled train subset, seed manifolds, and scalar regression head exactly. Do not accidentally substitute UTKFace's batch 32, gradient clipping 5, or other configs.

## Experimental design: four conditions, single trainer
Implement a common RSNA Plain/Uniform training path whose **only intended experimental factor** is timing/frequency and per-active-step strength of the supervised term. Define `U = len(unlabeled_loader)` and `L = len(labeled_loader)` **dynamically** for each seed, given the exact historical RSNA protocol; require `0<L<=U`. Each epoch consumes **one full unlabeled-loader traversal** and makes exactly **U optimizer steps**; always include the same `L_u` in each step.

- `front`: full-coefficient supervised MSE `L_s` only at optimizer indices `0,...,L-1`; labeled iterator consumed exactly once per epoch.
- `even`: full-coefficient `L_s` only at indices `floor((j+0.5)*U/L)` for `j=0,...,L-1`; labeled iterator consumed exactly once per epoch. This must match the UTKFace experiment's actual fixed-budget Even definition.
- `back`: full-coefficient `L_s` only at indices `U-L,...,U-1`; **pure** back clustering with no initial anchor; labeled iterator consumed exactly once per epoch.
- `dense_mass_matched`: every unlabeled optimizer step also receives `(L/U) * L_s`; recycle labeled batches on iterator exhaustion; total **nominal supervised coefficient mass per epoch** is L. Keep same U steps and same unlabeled objective.

Use `L_u = MSE(f_theta(strong(x_u)), stopgrad(f_theta(weak(x_u))))`, `r=1`, `lambda_u=1`; use labeled standardized-target MSE. No pseudo-label filtering, uncertainty weighting, teacher, probe, or extra optimizer updates. No extra supervised loss beyond the specified schedules. Do not alter or overwrite the historical files.

**Scientific caveat:** `front/even/back` isolate *placement* at fixed supervision count and coefficient; `dense_mass_matched` matches only **nominal coefficient mass**, NOT labeled-sample exposure or actual gradient magnitude. State this honestly in summaries. Do not claim the mass-matched condition isolates placement alone.

The same seed across four conditions must use the same manifest/cohort, initial model parameters, data augmentation definitions, optimizer and scheduler, and unlabeled batch ordering. Use independent per-role RNG/generators and verify where possible that additional labeled draws do not perturb the unlabeled stream. Freshly initialize the model for every run; record initial parameter SHA256. Each condition should be trained for 30 epochs with the same validation-only best-checkpoint selection.

## Exact run matrix
- Dataset: RSNA Bone Age, 5% labeled subset from existing manifests, **seeds 0,1,2,3,4,5**.
- Backbones: `resnet50`, `dinov2_vits14`.
- Conditions: `front`, `even`, `back`, `dense_mass_matched`.
- Total: **48** distinct run keys; each should be evaluated exactly once on the held-out test set, **after** validation-based checkpoint selection. No test-based tuning.
- Use the existing gpu04 RSNA data, existing official manifests, and pretrained model artifacts; **do not download datasets, regenerate splits, or access test labels before checkpoint selection**.
- Use a fresh date-stamped results root, e.g. `artifacts/rsna_anchoring_fixed_budget_20261117/`, with per-condition/backbone/seed subdirectories. Do not modify earlier artifacts, checkpoints, results, dirty research checkout, branches, or running jobs.

## Preflight required BEFORE large-scale queue launch
1. On **gpu04** check actual physical GPU inventory with `nvidia-smi`, GPU process occupancy and allocated VRAM, free disk quota, and access permissions. Do not kill, preempt, or compete with running jobs. Only use idle GPUs (up to 8), one worker per free GPU; if fewer than 8 are idle, start only a safe subset and record the constraint. If none are free, do not launch; report the blockage.
2. Check coherent source snapshot and required imports, libraries, pretrained weights, RSNA images, `cohort.json`, seed 0–5 ratio 0.05 manifests, and correct data paths in gpu04. Verify manifest digests and count `U,L` per seed from actual loaders; assert expected splits and no leakage. Avoid hard-coded UTKFace 563/29.
3. Implement automated CPU schedule-contract tests for all four conditions (unique steps, counts, index order and parity, mass equality). Verify in a synthetic no-data setting that every condition has U updates and U unlabeled terms; front/even/back have L supervision-containing updates; dense has U with coefficient L/U; and no extra optimizer updates.
4. Run an **isolated small GPU dry run** for **both** backbones (at most a short train/checkpoint smoke test) before launching 48 training jobs. Confirm one visible GPU in each process (`CUDA_VISIBLE_DEVICES`), correct `cuda:0` mapping, finite losses/gradients, enough VRAM, validation ability, and correct metrics schema. Avoid touching old outputs.
5. Check condition pairing and initial parameter hashes and confirm the compared trainer has identical optimizer/augmentation/model across conditions.

## Implementation and queue behavior
- Work in a **separate git worktree/branch** on gpu04; do not alter or clean/reset the dirty research checkout. Create new RSNA-specific trainer and launch/verification/report scripts, without changing historical scientific source. Record `git rev-parse HEAD`, trainer/helper SHA256s, data manifest digest, optimizer/batch config, GPU identifier, and environment/package versions per run.
- Use a restartable queue with explicit unique jobs and atomic state; a file lock / singleton controller; a separate log, stdout/stderr and PID for each worker; 8 concurrent workers only if 8 cards are available. Assign each process one physical GPU and `CUDA_VISIBLE_DEVICES` to that GPU, and validate the local CUDA device count is 1. Never share a GPU between jobs.
- Each job should refuse to overwrite completed results and verify `status`, source and manifest hashes, complete 30-epoch history, finite metrics, one best-checkpoint reload, exactly one held-out test pass, and correct seed/dataset/backbone/condition identity.
- If preflight passes, **start** the full queue and verify the first wave is genuinely running (live PIDs/processes and correctly bound GPUs), rather than merely generating a job list. A detached queue may be appropriate in the gpu04 shell; write its controller PID, per-GPU PID, queue state and logs, and commands for the user to inspect progress. Do not assert all 48 runs finished until they truly finish.
- Failed jobs should be marked failed with error/traceback and be safely retriable without overwriting successful jobs. No silent hyperparameter changes or selective exclusion of bad seeds.
- Respect shared hardware, privacy, and data usage permissions. No public publishing of patient-specific data or artifacts.

## Reporting and acceptance criteria
Save a complete `config.json`, `metadata.json`, `history.csv`, validation-selected best checkpoint, `metrics.json`, and ordered test predictions for each run (server local; not to public GitHub). At least save hashes and evidence for exact source/version and manifest identity.

On completed results, aggregate per-seed test MAE (months) and R2, means and sample/population SD (clearly labeled), and **paired** comparisons for front minus even, back minus even, dense_mass_matched minus even. Report 95% confidence intervals from n=6 seed-paired differences (e.g. t interval with df=5), with the same seed pairing for each backbone. Also report numbers of optimizer steps, supervised-containing updates, labeled batches consumed, total nominal coefficient mass, GPU times, and any incomplete runs.

Create:
- `README_RSNA_FIXED_BUDGET.md`: exact scientific design, config, source references, restart/monitor instructions;
- `RUN_MATRIX.json` or CSV, queue controller, script entrypoints;
- `PREFLIGHT_REPORT.md` with source/manifest/config and GPU occupancy evidence;
- `RESULTS_SUMMARY.md` with uncertainty if runs finished, or `QUEUE_STATUS.md` with observed live PIDs/first-wave status if still running;
- reproducible aggregation script and checks for paired n=6 results.

**Critical:** Reproduce the RSNA historical scientific recipe instead of quietly porting UTKFace optimizer/batch/clipping; keep the baseline source archive immutable. Return exact started-run count, active GPU IDs, process IDs, status and where outputs live. If blocked by unavailable resources, identify the precise blocker instead of pretending the experiment has started. Do not broaden scope or publish the draft research context publicly.
