# Repository Agent Instructions — Paper-only Cleanup Task

You are on a *staging* branch for a coding-agent cleanup request.

**Read [CODEX_PAPER_ONLY_CLEANUP.md](./CODEX_PAPER_ONLY_CLEANUP.md) completely before modifying files.** Its figure/table inventory and acceptance criteria define the task. The source of truth for historical as-run code is `essential_code/` with `essential_code/profiles.json` and `essential_code/SOURCE_INDEX.csv`.

Do not treat `essential_code/shared/` as one combined runnable snapshot. Do not modify the staging branch, historical source branch, `main`, or scientific behavior. Create a separate working branch, preserve source hashes, and produce the mapping/audit reports in the task specification.

**Privacy:** The manuscript is under anonymous peer review. The PDF is intentionally *not* stored on this public GitHub repository. Do not publish the manuscript, identify authors/submission, post public issues/PR descriptions about its identity, or infer missing manuscript details. If access to private PDF is necessary, obtain it privately from the user rather than fabricating paper coverage.

A task/instruction file does **not** mean an agent has already been started. Begin implementation only when the user invokes you in this repository/branch.
