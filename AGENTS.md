# ACTIVE Codex task — UCVME RSNA 5% fixed-budget 24-run campaign on gpu05 GPUs 4–7

Read [CODEX_UCVME_RSNA5_FIXED_BUDGET_24.md](./CODEX_UCVME_RSNA5_FIXED_BUDGET_24.md) **completely**. It is the **only active task** on this branch. Do **not** follow the older paper-cleanup task or relaunch any prior experiments.

The user authorized **24** UCVME fixed-budget scheduling runs:
- Host: `instgpu-05.cs.wisc.edu`, physical GPU IDs **4, 5, 6, 7 only**, if live verification finds those specific GPUs idle and VRAM-suitable (reported A40 48 GiB).
- RSNA Bone Age at **5% labels**, 2 backbones (R50/DINOv2 ViT-S/14), 4 conditions (FRONT, EVEN, BACK, DENSE-MASS-MATCHED), seeds 0–2, 30 epochs, fresh initial states.
- Preserve native **two-network UCVME heteroscedastic loss and stochastic consistency mechanism**. For supervision-absent steps, remove *both* the original labeled heteroscedastic losses and the labeled-only variance consistency term; keep original unlabeled uncertainty consistency intact. FRONT/EVEN/BACK have identical U iterations, L supervised positions, one labeled pass; DENSE has coefficient L/U at every U step, matches *nominal* supervised coefficient mass only.
- Audit the **as-run UCVME RSNA recipe**: do not import RSNA Plain's optimizer/batch/augmentation or alter precision/drop_last/sampler/uncertainty sampling settings silently. Use `essential_code/profiles.json` to recover coherent historical source. Require paired seed data/split and two-network initialization checks.

Perform safe CPU tests and genuine GPU preflight before starting a detached restartable queue. **If preflight passes, execute and confirm live worker PIDs**; if blocked, report precise blockers. Do not interrupt or use resources on gpu01–04 or other gpu05 GPUs; no unintended GPU oversubscription.

Keep experimental data/checkpoints/results/reviewer details off this public repo. Preserve old benchmarks and task branches. Report actual controller/PIDs/queue progress, accurate scientific differences and monitoring commands. A GitHub task file does not invoke Codex by itself.
