# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and GitHub
  issue/PR templates.
- Project metadata: `[dev]` optional-dependency group and centralized tool
  configuration (`ruff`, `pytest`, `coverage`, `mypy`) in `pyproject.toml`.
- CI: Python 3.9–3.12 test matrix, `mypy` type-check gate, coverage upload to
  Codecov, and a build/`twine check` job.

### Changed
- Version is now single-sourced from `src/drift/__init__.py` via Hatchling's
  dynamic version (no more hand-syncing `pyproject.toml`).

## [0.5.0] - 2025-06-04

### Added
- `embed()` — declarative batch embedding with cross-run dedup (MD5 hash scoped
  to `(model, sink)`), batching, exponential backoff, idempotent point IDs, and
  per-run cost tracking. `shadow_mode=True` runs with deterministic mock vectors
  and no API key.
- `watch()` — incremental CDC refresh over Delta Change Data Feed; handles
  insert / update_postimage / delete and writes the version watermark back to the
  ledger.
- `migrate()` — model upgrades via `dual-write` and `drift-adapter` (Orthogonal
  Procrustes) strategies, with an ARR ≥ 0.97 quality gate
  (`AdapterQualityError`).
- `Ledger` — SQLite lineage ledger (`~/.drift/ledger.db`) with `cost_by_model()`,
  `provenance()`, and `recent_runs()`.
- `drift` CLI: `embed`, `watch`, `migrate`, `status`.
- Qdrant and pgvector sinks (pgvector write-only for now).

[Unreleased]: https://github.com/aayush4vedi/drift-spark/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/aayush4vedi/drift-spark/releases/tag/v0.5.0
