# Codex Task — Five-node RTX 2080 Ti experiment launch

## Authorization, scope, and what success means

The user has explicitly authorized **starting** two focused semi-supervised regression experiment campaigns on their remote GPU machines. Perform the work **now**, subject to resource, dataset-license and preflight constraints. This is a new task, separate from the existing RSNA 48-run campaign; **do not interrupt or modify the running RSNA jobs**. You do not need to request another approval for the agreed experiments below.

This GitHub brief is a handoff for Codex. Simply creating this document **does not start an agent or an SSH session**. When Codex is invoked by the user, actually connect, perform checks, implement the minimal reproducible code, and, if preflight passes, launch the jobs and report their verified live status/PIDs. Do not report that jobs started before seeing running processes.

**Authorized SSH hosts** (user: \`enzez\`):
\`\`\`text
ssh enzez@instgpu-00.wisc.edu
ssh enzez@instgpu-01.wisc.edu
ssh enzez@instgpu-02.wisc.edu
ssh enzez@instgpu-03.wisc.edu
ssh enzez@instgpu-04.wisc.edu
\`\`\`
These are five hosts, each *possibly* with 8 RTX 2080 Ti GPUs. Inspect their actual hardware; do not infer that all 40 devices are present, idle, or suitable. On \`instgpu-04\`, an RSNA experiment is already running according to the user; **give that work absolute priority**, and do not use its GPUs or change any of its files. Never kill or preempt any other process.

**Target concurrency when 40 GPU slots are truly idle and usable:** 24 workers for Campaign A, 12 workers for Campaign B, 4 GPU slots reserved. If fewer devices are free, use fewer workers, preserve safe headroom, and report the real allocation. Maximum **one training process per physical GPU**, only on idle cards with enough available VRAM. Do not silently consume the 4 spare GPUs. Limit DataLoader workers and shared disk pressure appropriately. This is not permission to monopolize machines when others are using them.

Use the existing repository, shared dataset installations, model weights, virtual environments and manifests wherever valid. Use SSH keys/auth already available in the environment; do not ask for or log credentials. Avoid public repositories/issues for private experiment data. Do not push private data, checkpoints, hospital records or unpublished results to GitHub.

## Campaign A — UTKFace labeling-ratio × frequency/placement robustness

Scientific question: do supervised-anchor frequency and fixed-budget temporal-placement trends extend beyond 5% labeled UTKFace?

- Dataset: **UTKFace**, canonical existing cohort and split protocol.
- Labeled ratios: **0.01, 0.10, 0.20**.
- Backbone: **ResNet-50** and **DINOv2 ViT-S/14**.
- Paired seeds: **0, 1, 2** (first stage only; do NOT automatically expand to 3–5).
- Epochs: **30**.
- Objective: Plain/Uniform weak-to-strong consistency, **no uncertainty weighting, probes in the objective, or teacher terms**.
- Five frequency settings: **k = 1, 2, 4, 8, 20**, with anchor indices \`t % k == 0\` (t is zero-based); recycle labeled batches as required at anchor steps.
- Three *fixed-budget* placement settings: **front, even, back**, with one supervised pass per epoch, **no labeled recycling**. For each seed/ratio define \`U = len(unlabeled_loader)\` and \`L = len(labeled_loader)\` with the actual formal loader and \`drop_last\` conventions, and verify \`0 < L <= U\`. Exactly \`L\` supervised anchors per epoch:
  - front: \`0,...,L-1\`
  - even: \`floor((j + 0.5) * U/L)\`, \`j=0,...,L-1\`
  - back: \`U-L,...,U-1\` (pure back-loading, no early anchor).
- **Always exactly U optimizer steps and one complete unlabeled pass per epoch** across the 8 settings; unchanged \`L_u\` at every step. The only intended variation is the supervised-anchor schedule. At each active step, supervised coefficient = 1. Do not falsely claim that the *frequency* sweep holds labeled exposure fixed: it does not; only the front/even/back placement triplet does.
- Total **3 ratios × 8 settings × 2 backbones × 3 seeds = 144 training runs**.
- Full scientific objective: \`L_u = MSE(f_theta(strong(x_u)), stopgrad(f_theta(weak(x_u))))\`, \`r = 1\`, \`lambda_u = 1\`; labeled standardized-target supervised MSE when active.

Reference source locations, **read with profiles.json mapping**:
\`essential_code/shared/training/canonical_uniform_frequency.py\`,
\`essential_code/shared/training/supervised_spacing_decoupling.py\`,
\`essential_code/shared/scripts/run_canonical_uniform_frequency.py\`,
\`essential_code/shared/scripts/run_supervised_spacing_front_even_back.py\`,
\`data_processing/utkface_protocol.py\` and source-profile variants.

The legacy UTKFace modules are often **hard-coded to 5% labels, batch size 32, U=563, L=29**. Do NOT simply run these unchanged for 1% / 10% / 20%, and do not silently borrow incompatible snapshots across profiles. Build a single dedicated ratio-generalized, scientifically matched entrypoint with dynamic \`U,L\` and validated seed/ratio manifests. If a ratio manifest does not already exist, deterministically generate it using the exact existing \`generate_seed_manifest\` protocol and unchanged canonical cohort; record new manifests/digests and check nestedness and leakage. Do not regenerate or overwrite existing canonical manifests.

**Keep the as-run UTKFace Plain recipe**, including its batch size 32, transforms, full backbone fine-tuning, validation model selection, and optimizer/scheduler:
- ResNet-50: Adam lr 1e-4, weight decay 1e-3, StepLR(10, gamma=0.1);
- DINOv2-S: AdamW backbone lr 1e-5/head lr 1e-4, weight decay 0.05, CosineAnnealingLR(T_max=30);
- gradient clipping norm 5 for this UTKFace mechanism study; augmentation weak/labeled crop/flip and strong RandAugment per source protocol.
- Verify other model details (including dropout, data loader \`drop_last\`, worker seed generators) from the actual frozen reference rather than inventing them.

For paired conditions of a seed+ratio+backbone, ensure matching initialization (record an exact initial state hash), canonical train/validation/test membership and unlabeled data traversal. Use separate RNG streams where needed to prevent different numbers of labeled draws from changing the unlabeled stream.

## Campaign B — Supervised-only controls across four datasets

Scientific question: how much of the performance gain is explained by labeled exposure alone, and does unlabeled consistency add benefit relative to a compute-/exposure-matched supervised reference?

- Datasets: **UTKFace, IMDB-WIKI, RSNA Bone Age, CheXchoNet LVIDd**.
- Labeled ratio: **5%** (do not expand to other ratios).
- Backbones: **ResNet-50**, **DINOv2-S**.
- Seeds: **0,1,2** (first stage only).
- Modes:
  1. \`sup_only_one_pass\`: train only on labeled batches once per epoch (\`L\` supervised optimizer updates per epoch);
  2. \`sup_only_recycled\`: train using labeled batches repeatedly for \`U\` supervised optimizer updates per epoch, with \`U\` equal to the corresponding 5%-label unlabeled-pass length of the matched Plain protocol.
- Both modes have **NO unlabeled/pseudo-label loss**; do not train on pseudo-labels under a supervised-only label. Explain clearly that the two conditions do *not* have the same optimizer step count or labeled exposure. For \`sup_only_recycled\`, match the labeled exposure and optimizer update count of dense Plain/Uniform, within the same dataset/backbone/seed.
- **4 datasets × 2 modes × 2 backbones × 3 seeds = 48 training runs**.
- Use 30 epochs and exact dataset-specific canonical transforms, split/manifest, labeled target normalization, backbone initialization, optimizer/scheduler, \`drop_last\`, batch size, hyperparameters and lowest-validation-MAE checkpoint selection compatible with that dataset's original Plain comparison. **Do not blindly use UTKFace optimizer/batch/clipping for RSNA, CheXchoNet or IMDB-WIKI**.
- Reference supervised-only control: \`essential_code/shared/supervision_quantity_control.py\` (UTKFace only), \`essential_code/shared/training/supervised_benchmark.py\`, dataset-specific canonical Plain and protocol files and the source profiles in \`essential_code/profiles.json\`. Preserve the scientific distinction between a standardized general supervised benchmark and the exact Plain-matched supervised-only control. The latter is what this campaign is for.

CheXchoNet may be protected/restricted: first confirm valid authorized local access. Never copy restricted images, manifests with identifiable patient information, or outputs to an unauthorized host; if an approved host cannot access this dataset, mark that subset **blocked**, with the reason, rather than substituting another dataset or relocating protected data without permission.

## GPU/VRAM feasibility & scientific comparability

The RTX 2080 Ti typically provides ~11 GiB device memory. **Preflight FP32/full fine-tuning of BOTH backbones at the unchanged historical batch size on a single device**, using one short training iteration and minimal validation/checkpoint smoke test. Confirm model weights can be loaded on all selected hosts and that results are numerically finite. Do not downgrade batch size, image resolution, optimizer, precision, fine-tuned parameter set, augmentations or gradient-accumulation behavior silently to make a job fit: doing so would invalidate a claim of a directly matched protocol. If a configuration cannot fit, mark it blocked and report the exact OOM/availability evidence; run the remaining valid conditions.

Verify that existing gpu04 RSNA work and all other users' jobs are unaffected, check SSH host mapping, \`nvidia-smi\` GPU UUID/index, process occupancy, quota/disk free, Python/PyTorch/CUDA versions, installed dependencies, dataset manifests/cohorts and per-host access to shared files. Prefer safe local scratch for transient files and bounded number of DataLoader workers. Check test/train split disjointness and manifest SHA256; do not access test results during training.

## Distributed scheduling & provenance

Use a *separate experimental worktree/source path* and new results root distinct from all existing research runs, for example:
\`artifacts/ssl_controls_2080ti_20261117/{campaign_A,campaign_B}/...\`
under a directory/volume the five hosts can access according to the actual mount permissions.

**Do not assume the /nobackup filesystem or GPU job controller is shared across hosts.** Before execution, verify real paths and mount visibility. If shared locking cannot be established, split the immutable 192-job manifest into **disjoint per-host job partitions**, with never-overlapping output paths and a central human-readable index, so no two hosts can claim the same job. Record job placement and unique IDs \`campaign/dataset/ratio/backbone/seed/condition\`. Use an atomic queue state and a per-host worker lock / singleton controller. Do not spawn more workers than idle and authorized GPUs. Do not overwrite, clean, force-reset, remove or modify other projects' files; never auto-requeue a completed run.

Log for every run: exact source revision and trainer SHA256; source-profile references; all input manifest/cohort and checkpoint hashes; host, device UUID, PID, visible GPU mapping, elapsed time and VRAM; seed, ratio, labeled+unlabeled counts, \`L,U\`, per-epoch optimizer-step count, supervised-containing update count, labeled example exposure, and anchor indices or sampling schedule; full configuration; per-epoch training and validation history; validation-best checkpoint reload; held-out test inference **once after** validation selection; final MAE in original units and R2; ordered predictions local to restricted filesystem; completion/failure state. Use atomic file writes, no test-based hyperparameter tuning, and do not fail silently.

Required validation before launching all 192:
1. Assert **144 distinct A IDs** and **48 distinct B IDs** with no duplicate output paths.
2. CPU / synthetic schedule-contract tests for all three ratios and both styles, confirming \`U\` updates in A, exactly \`L\` supervision steps for fixed-budget schedules, \`ceil(U/k)\` for frequency, correct label iterator cycling, and untouched unlabeled objective.
3. Preflight 1 short run each backbone on actual **2080 Ti** without disturbing the historical environment. Check VRAM and dataset access/manifest integrity, and confirm the test loader was not accessed for model selection.
4. Confirm compatible and matched source/optimizer/augmentation for all four B datasets, and reveal differences if historical methods cannot be matched exactly.
5. Record all five hosts' actual GPU occupancy and safe available worker slots.

After preflight **launch the eligible workload** (do not merely produce code or scripts). Confirm controller and training PIDs are alive and GPU UUID affinity is correct, and record the *actual* counts active vs pending vs blocked. Use resume-safe job states and verify that already completed items are not run again. If SSH or authentication blocks execution, report the exact blocker with the relevant command result; do not fabricate a job launch.

## Output & concise report to user

Write:
- \`README_EXECUTION.md\` — how to resume/monitor/stop each host safely;
- \`RUN_MATRIX.json\` or CSV with all 192 immutable job keys, assigned slots and state;
- \`PREFLIGHT_REPORT.md\` including five-host occupancy/VRAM/source/manifest checks and OOM/permission failures;
- \`QUEUE_STATUS.md\` including **confirmed live** host, physical GPU UUID/index, PIDs, running and queued counts and exact results directory;
- \`aggregate_results.py\` producing per-seed test MAE and R2, means, SD and correctly paired differences/95% CIs for front vs even, back vs even and supervised-only recycled comparisons. Do not report aggregate results for incomplete seed triplets as if fully paired.

Present exactly:
(1) number of eligible, running, completed, blocked and pending jobs in A and B;
(2) GPU allocations by host, with proof no RSNA GPU was taken;
(3) exact monitor and logs paths;
(4) any deviations/limitations—especially OOM, inaccessible CheXchoNet, missing source/protocol, or unavailable manifests.

No fabricated results, no unannounced hyperparameter changes, no automatic expansion beyond 144+48, and no external/public release of experimental data.
