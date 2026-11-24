# ACTIVE Codex task — complete UCVME RSNA 5% to six seeds on gpu05

**Read [CODEX_UCVME_RSNA5_SEEDS3TO5_GPU05.md](./CODEX_UCVME_RSNA5_SEEDS3TO5_GPU05.md) IN FULL and follow it as the only active task.**

The user's new authorization is to launch **24 ADDITIONAL** RSNA 5% UCVME fixed-budget runs: backbones ResNet-50 and DINOv2 ViT-S/14 × schedules FRONT/EVEN/BACK/DENSE-MASS-MATCHED × **seeds 3,4,5**. The old 24 runs on seeds 0,1,2 are complete and **READ-ONLY**. Never relaunch them or modify their outputs. This instruction supersedes the old task's seed restriction without changing its native scientific recipe.

Original setup and strict scientific/safety requirements are documented in [CODEX_UCVME_RSNA5_FIXED_BUDGET_24.md](./CODEX_UCVME_RSNA5_FIXED_BUDGET_24.md). Keep the original full two-network UCVME native uncertainty objective and all loss components, RSNA 5% cohort, as-run source/optimizer, pretrained backbones, 30 epochs, paired initialization, and exact anchoring. Read existing frozen manifests and source on gpu05, not as a guessed combination of the repository archive.

Target **instgpu-05.cs.wisc.edu**, physically allowed GPU IDs **4–7 only**, using genuinely idle A40s after live verification. Never interrupt other work; use a fresh isolated run root and restartable queue. **If preflight passes, launch training and confirm real PIDs**, and after completion compute six-seed paired MAE uncertainty. No extra jobs or tuning. Keep all experimental output private on the cluster and report to the user **in English only**.

Writing these GitHub instructions does not start Codex or any remote execution.
