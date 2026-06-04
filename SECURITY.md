# Security Policy

## Supported versions

Drift is pre-1.0. Security fixes are applied to the latest released minor version
only.

| Version | Supported |
|---------|-----------|
| 0.5.x   | Yes       |
| < 0.5   | No        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use one of the following:

- GitHub's [private vulnerability reporting](https://github.com/aayush4vedi/drift-spark/security/advisories/new)
  (preferred), or
- email **4vedi.aayush@gmail.com** with the subject line `drift-spark security`.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected version(s) and environment.

You can expect an acknowledgement within **5 business days** and a remediation
plan or assessment within **30 days**. We'll credit reporters in the release
notes unless you ask us not to.

## Handling secrets and sensitive data

A few things to know about how Drift handles sensitive material:

- **API keys** (e.g. `OPENAI_API_KEY`) are read from the environment and are
  never written to the lineage ledger or logs. Never commit keys to the repo.
  Use `shadow_mode=True` for local development and CI — it needs no key.
- **The lineage ledger** (`~/.drift/ledger.db`) stores source-text hashes,
  model names, costs, and provenance — not raw document text or vectors. Treat
  it as you would any operational metadata store.
- **Sink credentials** (Qdrant, pgvector) are passed via connection URIs.
  Prefer environment-variable injection over hard-coding them in source.
