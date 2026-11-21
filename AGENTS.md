# Active Codex task — SAFE recovery of HPL factorial IMDB-WIKI core after reboot

**CLI LANGUAGE REQUIREMENT: Respond entirely in ENGLISH. This includes progress reports and final summaries. Do NOT switch to Chinese even though ChatGPT conversations may be in Chinese.**

Read [CODEX_HPL_IMDB_REBOOT_RECOVERY.md](./CODEX_HPL_IMDB_REBOOT_RECOVERY.md) **in full**. This is the only active task on this branch. Also read the original task at `tasks/hpl-factorial-schedule-vs-bilevel-20261120/CODEX_HPL_FACTORIAL_SCHEDULE_VS_BILEVEL.md` for exact run protocol, and the original private manifests.

User's previously authorized core factorial totals 48 runs: UTKFace 24 already valid; IMDB-WIKI 24 not completed after GPU host reboots at the last snapshot. Revalidate state before action. Resume interrupted cells only if full optimizer/RNG/data-stream continuation is valid; otherwise restart those cells from epoch 0 in new isolated attempt directories, without altering old partial results. Start truly queued cells only once. Never duplicate already-valid cells.

**AUTHORIZATION SCOPE:** Recover and finish ONLY the previously authorized 48-run Uniform/Full-HPL sparse/dense core. **Do not launch conditional P1 (no-BLO) runs or any newly designed study.** Validate live GPUs and protect all existing workloads. Keep original source/config/manifest/seed semantics, 30 epochs, val-best selection, one-time test evaluation. Track recovery transparently. Outputs remain PRIVATE on the lab hosts; public GitHub contains only instructions.

The mere GitHub branch update does NOT launch jobs; the user must send this branch to an active Codex CLI session.
