# Repository Guidelines

## Project Structure & Module Organization

The repository is both a Chinese interview handbook and a runnable Python companion. Numbered root files such as `12_Transformer与大模型原理.md` are the canonical chapters; keep `00_目录索引.md`, `README.md`, and related `[[WikiLinks]]` synchronized when chapters change. Python examples live in `code/chNN_topic/{core,llm,gpu}/`. Shared provider, environment, and compatibility helpers belong in `code/shared/`; automation belongs in `code/scripts/`; pytest coverage belongs in `code/tests/`. Architecture notes and implementation plans are under `docs/`. Docker assets are split between the root compose files and `docker/`.

## Build, Test, and Development Commands

Use Python 3.10 or newer. From `code/`:

```bash
python -m pip install -r requirements-core.txt
make ci-quick
make test
make lint
make ci
```

`ci-quick` validates chapters, README coverage, cross-references, and smoke behavior. `test` runs pytest without GPU cases; `lint` runs Ruff; `ci` additionally executes all core and mocked LLM examples. Use `LLM_MOCK=1 make test-llm` when no API key is available. Run a single example with `make run FILE=ch12_transformer_architecture/core/01_scaled_dot_product_attention.py`. GPU and real-API targets require explicit local dependencies and credentials.

## Coding Style & Naming Conventions

Python uses four-space indentation, Ruff formatting, a 110-character line target, and the rules in `code/pyproject.toml`. Name tests `test_*.py`, functions `test_*`, and chapter directories `chNN_snake_case`. Number examples in learning order (`01_...py`, `02_...py`). Each example should retain its `# ---` metadata block, run directly, use an `if __name__ == "__main__":` guard, and finish successfully with `OK`. Prefer readable teaching code over premature abstraction.

## Testing Guidelines

Pytest markers are `core`, `llm`, `gpu`, and `slow`; select the narrowest applicable tier. New shared helpers or bug fixes need focused tests in `code/tests/`. Missing optional dependencies should produce a clear `[SKIP]` result rather than a traceback. Before submitting, run `make ci-quick`, the relevant pytest file, and `make lint`.

## Commit & Pull Request Guidelines

Recent history favors imperative, scoped subjects such as `Fix verify.yml...`, `README: add...`, and versioned maintenance entries (`v1.0.13: ...`). Keep each commit focused and explain the affected chapter, tier, or workflow. Pull requests should summarize the change, list commands run, link an issue when applicable, and call out API, GPU, model-download, or Docker requirements. Include rendered screenshots only when Markdown diagrams or layout materially change.

## Security & Configuration

Never commit API keys, model weights, local caches, or `.env` files. Copy `.env.dockerexample` for local configuration and document any new secret in `docs/CI_SECRETS_SETUP.md`.
