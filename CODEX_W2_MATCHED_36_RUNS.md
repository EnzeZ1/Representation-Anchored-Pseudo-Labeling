# Codex task A — Launch 36 paired cross-dataset supervised-vs-SSL runs on gpu01

## Authorized action

The user has explicitly approved **launching these 36 fresh experiments now**. This document is only the handoff: GitHub changes DO NOT start jobs. When the user invokes Codex, inspect live remote resources, implement/audit the dedicated trainer, preflight, and launch a detached restartable queue **without asking again when checks pass**. Return verified GPU/PID/controller states, not a claim based on writing this file.

Scientific goal: for 5% labeled IMDB-WIKI, RSNA Bone Age and CheXchoNet, determine whether the weak-to-strong *unlabeled consistency* term improves performance over a strong recycled supervised-only control with exactly the same labeled supervision/update schedule. This is distinct from the existing 5% supervised one-pass/recycled Campaign B (48/48 complete), the completed UTKFace 1/10/20% controls (36/36), the 144-run UTKFace Campaign A, the earlier RSNA 5% experiment on gpu04, and the ongoing RSNA 10/20% campaign on gpu02 GPUs 4–5.

## Hardware and non-interference

- Work on **instgpu-01.cs.wisc.edu**, GPUs **0–7**, *provided they are still genuinely idle* at execution time. RTX 2080 Ti ~11 GB.
- Last user-provided snapshot at **2026-11-18 23:36 CST**: gpu01 0–7 idle (no jobs); gpu02 4–5 occupied with a separate RSNA extension; gpu02 0–3 and 6–7 reportedly idle; gpu03 occupied by root-owned services; gpu04 occupied by earlier RSNA. That snapshot is NOT a substitute for live nvidia-smi/PID/VRAM checks.
- Do not launch on gpu02/03/04 for this task, and do not touch any prior queues/worktrees/results. One trainer per verified free GPU, no oversubscription. Avoid concurrent source edits with the other independent Codex task.
- SSH: ssh enzez@instgpu-01.cs.wisc.edu.

## Exact 36-run matrix

- Datasets: **IMDB-WIKI**, **RSNA Bone Age**, **CheXchoNet LVIDd**, each at **5% labeled**.
- Backbones: **ResNet-50** and **DINOv2 ViT-S/14**.
- Seeds: **0,1,2**.
- Modes: **sup_only_recycled** and **dense_uniform_ssl**.
- Epochs: **30**.
- Count: 3 datasets × 2 backbones × 3 seeds × 2 modes = **36**, no extra seeds or tuning experiments.
- At ratio/seed-specific actual loaders define U = len(unlabeled_loader), L = len(labeled_loader); validate positive counts. All **36** runs use exactly **U optimizer updates per epoch**; all U steps have one coefficient-1 supervised standardized-target MSE on a recycled labeled batch. The modes differ only by whether the coefficient-1 Plain/Uniform weak-to-strong consistency term is active:
    - sup_only_recycled: L = L_s; never backpropagate any unlabeled loss.
    - dense_uniform_ssl: L = L_s + L_u, L_u = MSE(f_theta(strong(x_u)), stopgrad(f_theta(weak(x_u)))), sample weight r = 1, unlabeled coefficient lambda_u = 1.
- A step uses **one** optimizer update for the entire combined loss (no separate extra optimizer step for the supervised term). No uncertainty weighting, teacher, target filtering, frozen-probe penalty, or hidden gradient terms.
- Both modes must follow *the same independent labeled-batch sequence* for a matched seed/backbone/dataset, same initialization model-state SHA256, optimizer/groups, scheduler, data manifests/scalers, transforms (with relevant augmentation RNG accounting), dropout/model architecture/pretrained weights, validation-selected checkpoint protocol, full 30 epoch length, and labeled exposure/update count. Use role-isolated RNG streams where feasible, including separate unlabeled vs labeled generators; source/config/seed/initial model hashes recorded and equality asserted.
- IMPORTANT scientific nuance: a pure supervised-only run may perform no unlabeled model forward passes, whereas SSL performs them; model-dropout RNG and BatchNorm running-state updates can therefore differ. Audit this explicitly, and **do not misrepresent the two runs as differing solely in the gradient coefficient** if forward paths differ. Keep the requested two main modes as defined; do not silently turn sup_only_recycled into an unlabeled-BN-exposed placebo. If strict gradient-only isolation requires an additional lambda_u=0 forward-matched placebo, report that as a possible later experiment; **do not increase the approved 36 runs**.
- These comparisons quantify incremental benefit of the whole consistency-training branch at equal labeled exposure/updates, not a general proof that any unlabeled learning method always helps.

## Scientific source and as-run configurations

- Repo: EnzeZ1/Representation-Anchored-Pseudo-Labeling. The historical essential_code/profiles.json maps source snapshots to **separate** coherent profiles. Never blindly execute a mix of files from essential_code/shared. Existing gpu02 Campaign B and earlier plain+interaction results may use distinct initialization/RNG protocols; DO NOT claim existing B controls are strictly paired with new SSL unless equivalence is proved. This task instead permits fresh runs of BOTH conditions in a common dedicated implementation.
- Reference as-run Plain/Uniform regression with no reliability weighting; inspect actual original dataset/backbone baseline recipe and labels, seed-specific manifests/holdout cohort, augmented weak/strong views, normalized targets and best-val checkpoint semantics. Do **not** silently replace original method with a different algorithm.
- RSNA-specific: physical batch size **8**, two loader workers, drop_last=False; ResNet50 Adam encoder/head lr 1e-4/1e-3, zero weight decay, StepLR(10,0.1); DINOv2-S AdamW backbone/head 1e-5/1e-4, wd .05, cosine T_max=30; **no gradient clipping**; test MAE in **months**. The prior RSNA 5% 48-run source may be a useful read-only reference but is currently in use on gpu04 and must not be modified.
- For IMDB-WIKI and CheXchoNet: inspect their **actual frozen Plain/Uniform as-run** baseline (batch size, optimizer/head parameterization, clipping, drop_last, normalization, data license/paths) and reuse it without inventing "universal" UTKFace settings. Note HPL's distinct IMDB batch size 48 is NOT automatically the Plain/Uniform batch size. Test MAE in **years** (IMDB) and **cm** (CheX).
- UTKFace is **not** in this run matrix; do not redo completed controls.
- No use of any test labels for source choices, parameter selection, schedule choice, hyperparameter tuning or early stopping.

## Required preflight / launch

1. Remote login; record live GPU UUID, memory, process occupancy on gpu01, shell/venv/torch/CUDA, disk space and quota; inspect any existing 144-run controller to make sure the last experiment is truly finished or stopped and protect its files. If cards are occupied, use safe subset; if no cards free, report blockers.
2. Construct an explicit 36-key immutable run manifest and **separate new worktree/output root**, e.g. /nobackup/enzez/w2_matched_36_worktree_20261118/artifacts/w2_matched_36_20261118/. No overwrite of existing outputs, manifests or train artifacts. Store all results **locally**, not in public GitHub.
3. Audit split/manifest hashes, target scaler fitted only on labeled train, no leakage, source profile hashes, model initialization, batch/update counts and labeled stream equality. CPU/synthetic contract tests for 30×U steps / 30×U labeled batches each run; verify zero unlabeled-gradient in sup-only and correct stop-grad pseudo-target in SSL.
4. Preflight both backbones for all three datasets in FP32 at original per-dataset batch sizes, check finite loss/gradients, VRAM, loaders/transform permissions and checkpoint reload. Do not change scientific hyperparameters to accommodate an OOM. A blocked cell remains blocked with an error record.
5. Launch detached restartable queue across verified idle gpu01 GPUs 0–7, one trainer/GPU using CUDA_VISIBLE_DEVICES. Verify **real live** controller and worker PIDs plus successful logging. Do not change or kill any other experiments.
6. For every valid completed run persist config.json, environment/source/manifest/initial state hashes, run key, GPU/host, exact epochs/step and labeled exposure counts, history.csv, best validation checkpoint, final test metrics.json MAE/R2 and ordered predictions. Holdout test evaluated once **after** validation-only checkpoint choice and checkpoint reload.
7. Create PREFLIGHT_REPORT.md, RUN_MATRIX.csv, QUEUE_STATUS.md, READ_ME_EXECUTION.md, aggregate_results.py locally. Aggregate **paired** within same dataset/backbone/seed: delta = MAE(dense_uniform_ssl) − MAE(sup_only_recycled), with mean ± SD, explicit n and t-interval if n>=2; report exploratory n=3 uncertainty. Do not report unfinished jobs as complete or hide bad seeds.

## End-of-task report

Give live SSH-verified timestamp, actual GPU IDs used, launch count, 36-key completed/running/queued/failed/blocked, PIDs and tmux/controller, exact monitor commands and source/results paths. Say explicitly if source config or dataset limitations prevent comparable runs. No public posting of raw results, patient data, manuscript, reviewer information, or credentials. Keep the old 192/84/RSNA campaigns intact.
