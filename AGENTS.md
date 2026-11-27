# ACTIVE Codex task — six-seed 5% SSR placement across all methods and datasets

**Read [CODEX_SSR_ALL_METHODS_6SEED_PLACEMENT.md](./CODEX_SSR_ALL_METHODS_6SEED_PLACEMENT.md) COMPLETELY before any implementation.** This is the only active instruction on this branch. The user explicitly authorizes an idempotent FRONT/EVEN/BACK placement extension: six SSR methods, four datasets, two backbones, seeds 0–5, 5% labels, with scientifically verified existing run keys skipped.

When invoked in the user's authorized Codex CLI, FIRST audit cluster GPU occupancy, method provenance and existing artifacts, THEN implement and launch only eligible missing runs. Hardware: **idle physical GPUs on gpu01, gpu02, gpu03, gpu04**, plus **ONLY physical GPUs 6 and 7 on gpu05**. Leave all services, occupied/reserved GPUs and unrelated processes untouched; never preempt, restart or overwrite existing workloads. Do not touch gpu05 GPUs 0–5 or other servers.

Scientific fidelity: preserve each method's objective, native training/data protocol and paired split/initialization; this is NOT one universal Plain trainer. HPL bilevel/meta learning and UCVME two-network optimizers require explicit method-specific audits. Run matrix is 864 keys; reuse valid old cells, no duplicates, atomic cross-host queue, no additional ratios or DENSE schedules.

Make new private local worktrees and artifact paths. All CLI communication/progress reports in **English**. Keep private datasets, trained models, experimental numeric results, manuscript/reviewer information, credentials and cluster logs off the **public** GitHub repository. Report verified live worker PIDs/physical GPU UUIDs rather than treating this instruction file as evidence of running jobs.

**This GitHub branch by itself DOES NOT trigger Codex or start GPU jobs. The user must open it in an authorized Codex CLI session.**
