# Codex — active task: 84 new rebuttal runs on gpu03 and gpu04

**Read [CODEX_REBUTTAL_84_RUNS.md](./CODEX_REBUTTAL_84_RUNS.md) completely and execute that task now when invoked by the user.** The user has approved 48 RSNA Bone Age 10%/20% front/even/back/dense-mass-matched runs on gpu03 and 36 UTKFace 1%/10%/20% supervised-only one-pass/recycled controls on gpu04 (ResNet50 + DINOv2-S, seeds 0–2, 30 epochs).

Do not confuse this task with the previous paper-only code cleanup, RSNA 5% 48-run campaign, or the 192-run 2080 Ti campaigns on gpu01/gpu02. **Protect every existing run and all historical source/outputs**. First verify live GPU occupancy and that the older gpu04 RSNA controller is actually finished; use only idle GPUs and scientifically consistent manifests and settings. If preflight passes, launch the eligible workload in detached monitored queues and verify real PIDs and GPU UUID binding; do not pretend a GitHub instruction is an actual launch.

Do not publish scientific results, private datasets, or manuscript data to public GitHub. No silent hyperparameter, batch-size, precision, model or split changes. Report actual launch/status and monitoring paths.
