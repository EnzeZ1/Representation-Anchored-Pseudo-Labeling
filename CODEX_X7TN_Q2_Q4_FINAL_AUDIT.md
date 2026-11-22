# Codex: Final read-only audit for reviewer x7tN Q2 and Q4

**LANGUAGE REQUIREMENT:** All Codex CLI replies, progress updates, reports, and deliverables MUST be in ENGLISH. Do not switch to Chinese.

## Goal

The user asks whether x7tN **Q2** and **Q4** are actually finished. We have enough previous campaigns that lack of a consolidated report must NOT be confused with missing experiments. **Audit existing results now, and give a definitive, source-backed status for both questions. No new training or inference.**

### Exact reviewer questions
- **x7tN Q2:** "Do the frequency and placement conclusions hold at other label ratios (1%, 10%, 20%) and on another dataset?"
- **x7tN Q4:** "Please report supervised-only training (+/- recycling) for all datasets and backbones and the original methods in a FixMatch-style loop."

Do not treat these as merely checking job-count totals. For Q2, summarize the actual frequency and placement outcomes, including counterexamples. For Q4, distinguish genuine original native loops, faithful dense-loop adaptations, and supervised-only controls.

## Evidence already reported (verify, do not blindly inherit)

### Q2
- UTKFace 1%, 10%, 20% frequency/placement **144-run campaign** was originally assigned on gpu01 (task instructions in `tasks/ssl-2080ti-controls-20261117`; old controller may no longer be active). We have not yet received a definitive 144-cell final validity inventory or a results table. Inspect historical archives, manifests and actual source/output locations across authorized hosts. The anticipated design included frequency `k=1,2,4,8,20` and fixed-anchor FRONT/EVEN/BACK for each ratio/backbone/seed, but **derive actual run matrix, k schedules, anchor counts and budgets from the true manifest**, not these recollections.
- A separate **36/36** UTKFace 1%, 10%, 20% *supervised-only* ± recycling control is complete. This measures labeled exposure, not SSL frequency/placement; it cannot alone resolve Q2.
- RSNA Plain 5% fixed-budget **48/48 completed** (2 backbones, 4 schedules, 6 paired seeds). RSNA Plain 10%/20% extension **48/48 completed** (2 ratios, 2 backbones, 4 schedules, 3 paired seeds). Across 5% both backbones EVEN < FRONT/BACK on mean MAE; 10% DINO FRONT mean < EVEN, so no universal EVEN dominance. DENSE-MASS-MATCHED best mean in all these RSNA ratio/backbone blocks, but same nominal coefficient mass is NOT equal labeled example exposure. Prior 5% paired n=6 confidence intervals already reported.
- RSNA UCVME 5% same fixed-budget schedules **24/24 completed**, distinct objective (cross-method corroboration, not direct evidence of other label ratios).

### Q4
Q4 dense-loop 5% **six SSR methods x four datasets x two backbones x three seeds = 144 potential cells**, but do not assume all 144 are genuinely equivalent to a fully tuned / canonical official FixMatch-style loop. A previous archive audit found:
- Plain **24/24** live valid, dense Uniform;
- Mean Teacher **24/24** live valid native dense, not an extra independent added Ls;
- HPL **24/24** live valid adapted dense +Interaction;
- UCVME **24/24** live valid adapted dense +Interaction;
- SimRegMatch **5/24** live plus **18/24** completed archived, **1** CheX DINO seed 2 provenance unverified because earlier remote olvi access failed; treat `UNVERIFIED_ARCHIVE` separately from `MISSING`, do not launch a duplicate.
- Clean-dense Temporal Ensembling **24/24 COMPLETE_VALID as of Nov 22 at 10:56 CST** (UTK 7.1520±0.1648 R50 / 6.1660±0.1235 DINO; IMDB 10.8642±0.1383 / 9.8160±0.0886; RSNA 12.6888±0.0484 / 13.9129±0.8313; CheX 0.4623±0.0086 / 0.4675±0.0060). Confirm source/outputs; preserve seed 0,1,2 and population SD ddof=0.
- **Supervised-only ± recycling 5%** four datasets x two backbones x three seeds x two = **48/48 complete**; do not confuse with **36 matched Dense Uniform vs Sup-only Recycled** (three datasets x two backbones x 3 paired seeds x 2) which additionally supports unlabeled consistency beyond labeled repetition.
- Native vs dense caution: Official UTKFace HPL ALREADY traverses full unlabeled loader with labeled recycling, whereas paper short-loop HPL Original 59/59 is a locally modified benchmark adapter, not untouched upstream. Official IMDB HPL is labeled-controlled but its paper adapter differs in recipe. SimRegMatch percentile-scale issue needs concrete code provenance if it affects Q4.
- Q4 TE clean-dense **replaces** masked native supervised term with recycled clean labeled MSE each native step, while paper TE+Interaction **adds** a separate term. These are not identical variants. MT native dense is already labeled at every step. Compare with the official/FixMatch dense mixing **schedule concept**, not literal classification hard-label FixMatch losses.

## Task A — find the UTKFace 144-run campaign, classify every cell

1. Locate controller/manifest/archive(s) and executable as-run source SHA, on gpu01 and any relocated/archived authorized hosts. Verify actual planned cells and labels, ratios, backbones, seeds, k, placement anchor-budget and per-epoch U/optimizer-step counts. Check no duplicated/missing cells, stale statuses, interrupted-but-complete metrics or failed integrity guard. If old gpu01 is repurposed, search its archived output locations; do not conclude records are absent merely because nothing is currently RUNNING.
2. Produce **full 144-cell valid/failed/incomplete/unverified inventory** with absolute private paths and config hashes. Verify 30 epochs (or actual intended count) and val-best test once, finite metrics and histories, expected schedule/frequency/positions, seed pairing, labels/manifest, coefficient masses, no validation/test training leakage.
3. Compute **actual mean±population SD for every ratio × backbone × schedule** (all outcomes, no cherry-picking), plus paired differences when seed mapping is valid. Separate frequency sweeps (k) and placement fixed anchors. State which frequency/placement patterns replicate, which reverse and uncertainty n=3. **Do not claim 144/144 completion without checking.**
4. If any cells are missing or invalid, state exact cells and whether true archival uncertainty, without creating any new run or queue.

## Task B — consolidate six-method dense-loop audit for Q4

1. Validate exact 144 planned 5% method × dataset × backbone × seed cells, sources/versions and run metadata; do not duplicate completed runs. Include original benchmark schedules and explicit **supervised-containing / total** native steps where audited (UTK 5%: Plain 29/563; TE 465–497/593; MT 2252/2252; HPL 59/59 paper-adapted; UCVME 59/59; SimRegMatch 59/59). Distinguish paper adapter vs true official upstream.
2. Locate **one SimRegMatch archived CheXchoNet DINO seed 2 cell** and establish its provenance if possible via authorized read-only archives. Do not try to circumvent authentication or request credentials. If still unverified, label clearly and summarize aggregate only from honestly verified cells; do not silently fill seed data.
3. Produce concise tables: (i) one summary row per method: native supervision, dense-loop condition, valid cells and scientific comparability notes; (ii) original benchmark vs dense loop MAE results across all 4 datasets and 2 backbones where genuinely comparable, plus 5% supervised-only one-pass/recycled numbers; (iii) describe separately the six matched Dense Uniform vs Sup-only comparison blocks, no need to duplicate all underlying experimental numbers in main text.
4. Verify source/seed/protocol compatibility; some reported results come from different campaigns and are not strict paired A/B, don't attach paired intervals incorrectly. Explicitly flag where a dense-loop variant **is the same as +Interaction**, and where dense-loop is a **new clean control**. Flag HPL official-loop mismatch and SimRegMatch percentile issue as appropriate.
5. Distinguish "all 144 exact cells completed" from "we can answer the reviewer fairly"; do not claim an independently hyperparameter-tuned FixMatch-style dense method if the runs are fixed-parameter schedule adaptations. State whether remaining work is **just table synthesis**, **artifact provenance**, **genuine missing cells**, or **unanswered experimental demand**.

## Output (new private local folder only)

Write in a separate newly created audit folder on the authorized lab host, never overwriting previous outputs:
- `X7TN_Q2_UTK144_INVENTORY.csv` (cell-level)
- `X7TN_Q2_RATIO_SCHEDULE_RESULTS.md` (complete result tables, sign/uncertainty/counterexamples)
- `X7TN_Q4_SIX_METHOD_DENSE_INVENTORY.csv` (cell-level)
- `X7TN_Q4_DENSE_COMPARISON.md` (matched/provenance caveats and tables)
- `X7TN_Q2_Q4_FINAL_VERDICT.md`: For each Q, exact status **FULLY ANSWERABLE / SUBSTANTIALLY ANSWERABLE WITH CAVEATS / NOT YET ANSWERABLE**, and minimum missing information. Provide a short English rebuttal-ready paragraph for each **only if data support it**. Avoid exposing unpublished numbers in public GitHub.

If existing past audit reports help, use them, but validate any novel numerical claim from actual artifacts.

## Strict boundaries

**READ ONLY. Do not start, schedule, resume, rerun, or alter any experiment, including the live HPL IMDB factorial 15 jobs. No GPU test inference, checkpoint edits, controller changes, preemption, status rewrites, data modifications, or manuscript modifications.** CPU-only inspection/reporting in a fresh private folder is permitted. Do not interact with service GPUs. Do not upload manuscript/rebuttal, results, raw patient data, logs, private paths identifying patients, reviewer identities or credentials to this public GitHub repository. All communication and reports in ENGLISH.

## Required CLI answer

Reply with:
1. Actual UTK 144 planned/valid/missing/unverified counts and schedule/ratio outcome summary.
2. Q4 six-method valid count by method, especially SimRegMatch CheX DINO seed 2 provenance, plus TE complete status.
3. Precise verdict separately for x7tN Q2 and Q4, with remaining gaps (if any).
4. Absolute private local report paths and evidence source/commit/checkpoint details.
5. Confirmation that no training or existing results were changed.

**Remember: the GitHub file alone does not execute Codex.**
