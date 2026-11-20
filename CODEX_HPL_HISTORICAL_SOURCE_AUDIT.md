# Codex: READ-ONLY historical HPL source / experiment audit — do NOT launch new runs

## User request and outcome

The user recalls an **older Codex modification of HPL official code**, and believes that a **"without our intervention"** experiment may already have been run. Before recommending or launching the proposed `sparse-extended` 12-run experiment, **investigate historical code and existing results** so we do not repeat work or misdescribe HPL's native labeled recycling.

**This is an audit only, not authorization to train, stop, restart, requeue, or otherwise change any experiment.** GitHub contains task instructions only; files pushed here do not run Codex. The user will invoke this handoff through the live Codex session.

Priority: exact HPL as-run provenance on **UTKFace and IMDB-WIKI at 5%**, R50 and DINOv2-S, seeds 0,1,2 (expand to other seeds/datasets for discovery, not extra work), and comparison of **official upstream HPL**, **our benchmark "Original HPL"**, **our benchmark HPL+Interaction**, and **any earlier no-recycling / short-loop / full-unlabeled sparse-control experiments**.

## Existing known evidence — CHECK rather than assume

- Repo: `EnzeZ1/Representation-Anchored-Pseudo-Labeling`. Preserved historical source records: `essential_code/profiles.json`, `essential_code/SOURCE_INDEX.csv`, and versioned source files; **essential_code/shared is not necessarily one coherent executable snapshot**. Historical module `essential_code/shared/third_party/Heteroscedastic-Pseudo-Labels/utkface/main_ours.py`, original upstream lineage recorded as `sxq11/Heteroscedastic-Pseudo-Labels@89f9f8bd` (independently verify actual complete commit). Other original HPL adapters for medical data may differ.
- In one preserved benchmark code snapshot, `--ssl_mult=2` *duplicates labeled examples at dataset level* for `training_schedule=original_hpl`, and `run_epoch` uses `total_itr_num=len(dataloader_lb)`. Main predictor loss already contains `L_s + w_ulb * L_u^{HPL}` **at every executed predictor iteration**. Meta/uncertainty learner has additional label-dependent updates (every fifth predictor iteration). Dataset duplication is a kind of repeated labeled exposure, but NOT automatically the same thing as cycling the iterator for U full unlabeled steps.
- In `training_schedule=label_recycling` benchmark variant, `labeled_repeat=1`, `total_itr_num=max(len(labeled_loader),len(unlabeled_loader))` and labeled iterator is restarted on exhaustion. This extends **both** the total main predictor updates and the full unlabeled traversal; it does NOT simply add `L_s` to previously unlabeled-only steps.
- A previous verified loop audit reported UTKFace 5% HPL original **59 supervised-containing iterations out of 59 total** (with labeled dataset multiplication); IMDB-WIKI original **598/598**. Contrast `U=563` UTKFace and ~5685 IMDB-WIKI, but use **as-run actual config/loader counts**, not these estimates as immutable facts.
- Previous paper result aggregates show UTKFace HPL +Interaction often improves MAE while IMDB-WIKI HPL+Interaction may worsen. This is the *reason* we want to identify actual protocol differences, not an instruction to select experiments/post-hoc seeds.

**Potentially misleading shorthand to avoid:** "Official HPL doesn't recycle labels"; "Our method is the first to recycle labeled batches in HPL"; "HPL original uses no labeled supervision"; "Both HPL variants use equal optimizer step budgets." None is grounded without source/run-specific qualification.

## Step 1 — Recover old as-run trainer and exact Codex modifications (READ ONLY)

1. Identify the as-run source and complete SHA hashes for each HPL historical **Official/native**, **Original regression benchmark**, **+Interaction/reorient**, and any other older variant from the git history, preserved source profiles, local worktrees, run config JSONs, controller launch scripts and archived outputs. Search older commits/branches, tagged scripts, run directories, `git log --all --follow`, `git diff`, historical change records and any previous Codex-generated code. A single current `main_ours.py` version is NOT sufficient to assert what earlier experiments executed.
2. Compare **upstream pinned source (verified commit)** vs **local regression port** using minimal relevant code diffs and line-numbered evidence: labeled dataset construction and `ssl_mult`, dataloader `drop_last`, `run_epoch` iteration count and labeled iterator exhaustion behavior, predictor update loss and optimizer calls, uncertainty weighting, `higher` inner-loop/meta updates and per-5-step cadence, auxiliary uncertainty learner optimizer, initialization/augmentation and evaluation protocol, batch size and epochs.
3. Identify which changes Codex made historically, if git provenance genuinely supports attribution. If provenance cannot prove who made them or when, state "source differed" instead of speculating about authorship. Distinguish official algorithm design from our official-code-derived **regression adaptation**, and benchmark choices.
4. Check whether differences depend on **dataset/backbone**, especially HPL UTKFace vs IMDB-WIKI, including `w_ulb`, `lambda2`, `ssl_mult`, native iterator length L, U, dropout, head/encoder optimizers, seed-specific labeled manifests and checkpoint selection. Do not mistake differing HPL implementation revisions for an effect of the schedule.

## Step 2 — Find whether the remembered "without intervention" experiments already exist

Search actual local/persisted and archived **HPL experiment artifacts** first, including older names such as `original_hpl`, `original`, `canonical`, `no_reorient`, `label_recycling`, `reorient`, `anchor_interval`, `sparse`, `timing`, `no_recycle`, and older Codex experiment output directories.

- Audit all discoverable UTKFace/IMDB-WIKI 5% HPL results, R50 and DINO seeds 0–2 (if six seeds present, index them all but do not expand this task). Inspect `config.json`, launch args, main trainer SHA / git commit, method identifier, training and validation logs, actual per-epoch main optimizer steps and supervised-containing anchors, `ssl_mult`, labeled example exposure, unlabeled batches covered, auxiliary HPL meta cadence and optimizer counts; checkpoint/validation-best/test-once proof; `metrics.json`, seed/manifest/protocol identity.
- Classify each **completed valid source-backed run** into distinct conditions:
  **A**. Official native HPL or its benchmark Original: short labeled-controlled loop, `ssl_mult` may duplicate labeled data, all native predictor steps supervised.
  **B**. HPL +Interaction: full U-step unlabeled traversal and supervised predictor branch at each step, repeated labeled batches.
  **C**. **Sparse-extended control**: full U-step unlabeled traversal but only the original L_native (e.g. 59) supervised *main predictor* anchors, no direct supervised predictor loss at other steps, while original HPL meta/uncertainty learner behavior remains explicit. Note this is NOT truly label-free even on nonanchor steps if meta updates use labeled data.
  **D**. Other variants (name the exact semantics; e.g. forced full-U but different `ssl_mult`, unsupervised-only, altered meta cadence or total epoch count).
- IMPORTANT: Do not confuse **A vs B** (existing paper Original/+Interaction) with **C vs B** (the controlled 563-step sparse-vs-dense experiment proposed in our recent conversation). The user remembers an older "without this" experiment; determine which category it actually was from executed artifacts, not names.
- Clearly answer: *Did we already perform C on UTKFace? On IMDB-WIKI? For which backbone/seeds, with which version and verified metrics?* If yes, **do not recommend rerunning**. If no, report precisely what is missing and whether existing A/B runs are even scientifically paired.
- Identify potential old runs on remote archived hosts, but access SSH **read-only**; failed SSH or lost archival file should be marked `UNVERIFIED_ARCHIVE`, not `MISSING`. Do not ask user to expose credentials. Prefer already mounted project dirs or servers authorized to the user; never commandeer GPUs to investigate.

## Step 3 — Deliver a concise answer with evidence, not another experimental campaign

Produce private local reports (new, distinct analysis/report folder, no edits to prior run artifacts), e.g.:

- `HPL_OFFICIAL_VS_ADAPTED_SOURCE_DIFF.md`: exact official commit, old/new source paths+hashes, code excerpts for labeled duplication/recycling, optimizer-step budget, meta cadence; which code was actually used.
- `HPL_EXISTING_RUNS_INVENTORY.csv`: dataset/backbone/seed, classification A/B/C/D, path/config/source hash, valid status, U/L, `ssl_mult`, predictors/meta updates, epochs, MAE, whether paired provenance holds. Report all available results; don't cherry-pick.
- `HPL_W1_RESPONSE_AND_GAP.md`: a **short English rebuttal draft** answering eQjg W1 in a defensible way, plus whether the proposed sparse-extended 12 runs are redundant and exact gaps, if any.

If A and B are existing but C absent, **just recommend** the smallest necessary C-only experiment (UTKFace + IMDB-WIKI, two backbones, three predeclared seeds = max 12) and separate scientific confounds. Do NOT create/launch/schedule it without further user authorization.

User-facing Codex response must explicitly answer in Chinese with:
1. **Official upstream HPL truly does what?** (labeled oversampling/iterator behavior; exact cited source).
2. **What did the old Codex adaptation change?** (dataset duplication, loop length, recycling, meta/labeled loss; as-run grounded).
3. **Which "without" runs are already complete/valid for UTKFace and IMDB-WIKI?** Table of A/B/C by backbone/seed, concrete paths if available.
4. **Should we run any more?** Only if a genuine condition C gap remains; no unchecked claim that Original HPL lacks recycling.

## Strict non-interference / privacy

**NO new training, GPU inference, test inference, GPU preflight, experiment queue changes, source edits, controller restarts, checkpoints edits, status reclassification, or rescoring test via model.** CPU-only checks of existing logs/results permitted; no private results on public GitHub. Do not interfere with W2, Q4 TE, MT/TE, RSNA, UCVME or any existing HPL experiments. A source audit is successful even if it concludes the proposed experiment has already been done.
