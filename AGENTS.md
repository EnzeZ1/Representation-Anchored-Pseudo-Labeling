# ACTIVE Codex task — READ-ONLY RSNA 5% three-seed re-reporting

Read **[CODEX_RSNA_PLAIN_UCVME_3SEED_READONLY_REPORT.md](./CODEX_RSNA_PLAIN_UCVME_3SEED_READONLY_REPORT.md)** completely; it is the only active task on this branch.

**Do not train anything or execute GPU inference.** The user requests a CPU-only report from already-completed artifacts:
- Plain RSNA 5% original campaign on gpu04: use **only seeds 0,1,2** to produce an n=3 subset summary; keep original seeds 0–5 (n=6) untouched and report the six-seed statistics separately as a reference.
- UCVME RSNA 5% completed campaign on gpu05: use its existing seeds **0,1,2** (n=3), retaining its native two-network objective and original output.
- For each method/backbone, report FRONT, EVEN, BACK and DENSE-MASS-MATCHED mean±population SD, per-seed data, paired differences and exploratory 95% paired CIs. Check all provenance using existing files. Clearly distinguish FRONT/EVEN/BACK (matched labeled exposure) from DENSE (coefficient-mass matched but greater exposure).
- No new seeds, no queue launch, no changes to original artifacts, no repeated test evaluation, no publication of private research metrics or patient data in GitHub.
- The previously created UCVME seeds 3–5 branch was CANCELLED. Do not revive it.

**Respond entirely in English.** State exactly which already-existing artifacts were read and what you could verify; publish any detailed result report only privately on the authorized remote systems.

The creation of this GitHub task does not trigger Codex or remote execution.
