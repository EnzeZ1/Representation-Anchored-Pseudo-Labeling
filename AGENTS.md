# Active Codex task — HPL official-baseline FAIRNESS AUDIT (READ-ONLY)

**LANGUAGE REQUIREMENT: Reply in ENGLISH in the CLI, including progress messages, explanations, reports, and the final summary. Do not switch to Chinese.**

Read [CODEX_HPL_OFFICIAL_BASELINE_FAIRNESS_AUDIT.md](./CODEX_HPL_OFFICIAL_BASELINE_FAIRNESS_AUDIT.md) in full and follow it as the sole active task instruction on this branch. This task is specifically about official upstream HPL training-loop fidelity, old Codex modifications, and whether prior valid experiment artifacts can serve as faithful, matched official-baseline results.

Prior history audit reports are on gpu04 at `artifacts/hpl_historical_source_audit_20261120_130602/`; inspect those as background, then verify all consequential claims from pinned upstream source, as-run configurations and logs.

**No training or inference is authorized.** Do not start, schedule, stop, resume, or requeue experiments. Do not change source/configs/checkpoints/results. Any proposed new experiments are a plan for user approval only. Do not interfere with ongoing rebuttal campaigns. Do not publish private results or reviewer information to this public GitHub repository. GitHub instructions do not automatically invoke a Codex agent.

Final deliverables must say whether suitable faithful upstream-loop results already exist for UTKFace and IMDB-WIKI, whether old HPL k=1 runs can stand in, the precise fairness issue, and the minimal next step, with source-backed evidence.
