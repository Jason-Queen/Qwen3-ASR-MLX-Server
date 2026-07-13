# AGENTS.md

<!-- assembly-provenance: {"canon_revision":"sha256:0f62c4be8b4090488f4aa63f9aca1c0285142b8bdd7b0225d794877fef7e6877","contract_revision":"sha256:88d17a8a34b26d06a44ed4cc0aa662040f893828be208031be38527cf33e0ff7","harness_revision":"sha256:10f15264ac63d74ad2a04f80cdcb66cbc9de275d4a21dfde6b9d22e0cee8dc7c","repo_facts_revision":"sha256:377ed2fd503aa66d7e2808006e3011f94827c27a6c75b64a1e38e0599fc75374","schema":1,"target":"agents-cross-agent"} -->

## Surface And Scope

- Primary runtime surface: Cross-agent baseline.
- Supported use cases: repository-wide implementation, testing, packaging, branch integration, and release verification for the Qwen3 ASR MLX server.
- Out-of-scope use cases: machine-specific launchd ownership, local model inventory, temporary task briefs, and unlabeled runtime-specific controls.
- Keep machine paths and workspace service operations in the parent workspace documentation. Keep this file portable and version it with the code.

## Repo Map

- Top-level directories discovered during inspection: `scripts`, `tests`, and generated `qwen3_asr_mlx_server.egg-info`; the generated metadata is not source of truth.
- Key manifests discovered at the repo root: `Makefile`, `pyproject.toml`, `uv.lock`, and `requirements.txt`.
- Existing instruction files to reconcile: `AGENTS.md`.
- `whisper_mlx_server.py` implements the API and inference pipeline. `qwen3_asr_mlx_server.py` is the CLI and startup entry point.
- `tests/` covers endpoint behavior, language handling, model resolution, and inference serialization. `scripts/smoke_local_api.sh` exercises the live HTTP contract.

## Workflow And Handoffs

- Repair versus new-file context: repair this existing shared repository contract in place; do not create branch-specific copies.
- Nested context or monorepo signals: none detected. The parent local workspace may add machine-specific instructions without changing this portable contract.
- Create development branches from `main`. In the local two-worktree layout, `debug/local` is the stable integration branch; routine promotion uses a normal merge, preferably `--ff-only`. Use cherry-pick only for an intentional backport.
- Keep each commit isolated and complete. Add or update tests when behavior changes, and keep the README, package version, lockfile, API version, and smoke behavior consistent.
- Handoff checkpoints: record the branch and commit, modified files, verification status, live-service state when relevant, and remaining work.
- Intent-bearing handoff: record purpose, rationale, material tradeoffs, risks, deviations, and validation evidence; a file-by-file summary is only a navigation aid.
- Intermediate work needs an owner and reconciliation path. The integrator reviews the final diff and reruns the release gates before calling the change complete.

## Commands And Verification

- Explicit repo commands: `make sync`, `make test`, `make run`, `make smoke`.
- Confirmed repo-standard test command: `make test`.
- The Make targets delegate to these direct commands:
  - Install: `uv sync --python 3.11`
  - Unit tests: `uv run python -m unittest discover -s tests`
  - Start: `uv run python qwen3_asr_mlx_server.py`
  - Smoke: `./scripts/smoke_local_api.sh 8989`
- Verification gates: run the unit suite and exercise every affected endpoint with the smoke script before integration. A documentation-only change still requires syntax and link/path review.
- Preserve these live contracts: `/healthz`, `/v1/models`, multipart and JSON transcription, Chat Completions audio input, supported response formats, and `task=transcribe` rejection boundaries.
- For changes that can affect MLX execution, verify serial inference locking, temporary-file cleanup, model fallback behavior, and timestamp/language metadata.

## Constraints And Boundaries

| State-changing operation | Enforcement surface | Confirmation and verification path |
| --- | --- | --- |
| Change API or inference behavior | Implementation, tests, and smoke script in the same branch. | Review the diff, run the unit suite, then smoke the affected endpoints against that checkout. |
| Change dependencies, versions, or release metadata | `pyproject.toml`, `uv.lock`, README, and the FastAPI version must describe one release state. | Inspect all four surfaces, run `uv sync --locked`, and rerun tests before integration. |
| Download models or widen network exposure | Setup is explicit; runtime defaults to `127.0.0.1` and Qwen3 ASR model aliases. | Confirm the requested model/source and resolved path. Never treat `0.0.0.0` as a production-safe default. |
| Integrate or publish code | Work on a branch rooted in `main`; keep local `debug/*` branches unpublished unless an owner explicitly changes that policy. | Confirm a common merge base, review `main...<branch>`, integrate normally, rerun release checks, and push only the intended release branch or tag. |
