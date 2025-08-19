# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Application code. Entry point is `src/main.py`. Lesson modules live under `src/week2/` and `src/week3/`. Data and images are in `src/assets/`.
- `tests/`: Pytest-based tests (e.g., `tests/test_main.py`).
- Tooling: `pyproject.toml` (deps, Poe tasks), `ruff.toml` (lint/format), `mypy.ini` (types), `.pre-commit-config.yaml` (hooks).

Example layout:
```
src/
  main.py
  week2/, week3/
  assets/
tests/
```

## Build, Test, and Development Commands

- Install deps: `uv sync`
- Run app: `uv run poe run` (alias for `python src/main.py`)
- Run tests: `uv run poe test`
- Tests + coverage: `uv run poe test-cov`
- Lint: `uv run poe lint` (Ruff checks)
- Format: `uv run poe format` (Ruff formatter)
- Type-check: `uv run poe type-check` (Mypy)
- All checks: `uv run poe all-checks`
- Pre-commit (manual): `uv run poe pre-commit`

## Coding Style & Naming Conventions

- Language: Python 3.13 (`.python-version`), 4-space indent.
- Style: Ruff-managed; line length 100; double quotes; imports sorted with `src` as first-party.
- Types: Strict Mypy; add explicit type hints.
- Naming: modules/packages `snake_case`, classes `CapWords`, functions/vars `snake_case`, tests `test_*.py` with `test_*` functions.

## Testing Guidelines

- Framework: Pytest with `pytest-cov`.
- Location: put tests in `tests/` mirroring `src/` structure.
- Conventions: name files `test_*.py`; keep tests fast and deterministic. Use fixtures/`monkeypatch` for IO or state (see `tests/test_main.py` capturing stdout).
- Run locally: `uv run poe test` or `uv run poe test-cov`.

## Commit & Pull Request Guidelines

- Commits: follow Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `test:`, `chore:`). History shows `feat: ...` in use.
- Scope: small, focused commits; include rationale when non-trivial.
- PRs: provide a clear description, linked issues, before/after output or screenshots when user-visible. Note any follow-ups.
- Quality gate: ensure `uv run poe all-checks` passes; update docs/tests when behavior changes.
- Assets: avoid committing large binaries; place small data under `src/assets/`.

## Security & Configuration Tips

- Do not commit secrets or tokens; use environment variables/local config.
- Install hooks: `uv run pre-commit install` to enforce checks before committing.
