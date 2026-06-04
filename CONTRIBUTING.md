# Contributing to Drift

Thanks for your interest in improving Drift. This guide covers how to get a dev
environment running, the conventions we follow, and what a good pull request
looks like.

## Development setup

```bash
git clone https://github.com/aayush4vedi/drift-spark
cd drift-spark
python -m venv .venv && source .venv/bin/activate
pip install -e '.[spark,qdrant,pgvector,dev]'
```

The `dev` extra pulls in everything you need to lint, type-check, and test.

## The local check loop

Run these before opening a PR — they mirror what CI enforces:

```bash
ruff check src/ tests/ examples/ integration-tests/   # lint
ruff format src/ tests/                                # format
mypy                                                   # type-check (src/drift)
pytest tests/ --cov=drift --cov-report=term-missing    # unit tests + coverage
```

Unit tests need no Docker and no API key — Spark sessions and DataFrames are
mocked.

### Integration tests (optional, maintainer-oriented)

These exercise the real Spark / Delta / Qdrant paths and require local
infrastructure:

```bash
docker run -p 6333:6333 qdrant/qdrant
python integration-tests/e2e_smoke_test.py
python integration-tests/it-adapter.py
python integration-tests/it-migrate.py
```

## Conventions

- **Style & linting:** `ruff` (config in `pyproject.toml`). No manual style
  debates — let the formatter decide.
- **Types:** new code in `src/drift` must pass `mypy`. The package ships
  `py.typed`, so the public surface is expected to be fully typed.
- **Tests:** new behavior needs a test. Public API changes need coverage. Keep
  unit tests free of network/Docker/API-key dependencies (use `shadow_mode`).
- **Commits:** we follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`). Keep commits
  focused and atomic.
- **Changelog:** user-visible changes go under `## [Unreleased]` in
  `CHANGELOG.md`.

## Pull requests

1. Fork and branch off `main` (e.g. `feat/pgvector-cdc-delete`).
2. Make the change, add tests, run the local check loop.
3. Update `README.md` / `CHANGELOG.md` if behavior or APIs changed.
4. Open the PR using the template; describe the motivation and link any issue.

Small, reviewable PRs get merged faster than large ones. If you're planning a
big change (new sink, new provider, new migration strategy), open an issue first
so we can agree on the shape before you write the code.

## Adding a sink or provider

- **Sinks** live in `src/drift/embed.py` (`_upsert_*` / URI parsing). A new sink
  should support the write path and, ideally, the CDC delete path used by
  `watch()`.
- **Providers** are parsed from the `"provider/model-name"` string. A new
  provider needs an embedding call, a cost entry, and a `shadow_mode` path.

## Reporting bugs and requesting features

Use the GitHub issue templates. For security issues, **do not** open a public
issue — see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
