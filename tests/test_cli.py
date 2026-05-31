"""CLI smoke tests — no Spark, no Docker, no API keys needed."""

import re

from typer.testing import CliRunner

from drift.cli import app

# CI has no TTY but Typer still emits ANSI in help text; strip before substring checks.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner(env={"NO_COLOR": "1"})


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


# ── help text ────────────────────────────────────────────────────────────────

def test_help_exits_zero_and_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("embed", "watch", "status", "migrate"):
        assert cmd in result.output


def test_embed_help_shows_key_flags():
    result = runner.invoke(app, ["embed", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    for flag in ("--table", "--text-col", "--model", "--sink", "--shadow-mode", "--no-dedup"):
        assert flag in out


def test_watch_help_shows_key_flags():
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    for flag in ("--table", "--text-col", "--sink", "--since-version", "--shadow-mode"):
        assert flag in out


def test_status_help_shows_sink_and_limit():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--sink" in out
    assert "--limit" in out


def test_migrate_help_shows_from_to_strategy():
    result = runner.invoke(app, ["migrate", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    for flag in ("--from", "--to", "--sink", "--strategy"):
        assert flag in out


# ── drift status — real ledger smoke test ────────────────────────────────────

def test_status_no_runs_for_unknown_sink():
    # Uses default ledger at ~/.drift/ledger.db (created if absent).
    # The unique URI guarantees no prior runs exist for this sink.
    result = runner.invoke(app, [
        "status", "--sink", "qdrant://ci-smoke-test/no-such-collection"
    ])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_status_with_seeded_ledger(tmp_path):
    from unittest.mock import patch
    from drift.ledger import Ledger
    from drift.embed import EmbedRun

    ledger = Ledger(db_path=tmp_path / "test.db")
    run = EmbedRun(model="openai/text-embedding-3-small",
                   sink="qdrant://localhost/col",
                   n_rows_processed=5, n_rows_deduped=2, cost_usd=0.0001)
    ledger.write_run(run)

    with patch("drift.ledger.Ledger", return_value=ledger):
        result = runner.invoke(app, ["status", "--sink", "qdrant://localhost/col"])

    assert result.exit_code == 0
    assert "openai/text-embedding-3-small" in result.output
    assert "rows=5" in result.output


# ── drift migrate — stub behaviour ──────────────────────────────────────────

def test_migrate_dual_write_exits_one_with_stub_message():
    result = runner.invoke(app, [
        "migrate",
        "--from", "openai/text-embedding-3-small",
        "--to",   "openai/text-embedding-3-large",
        "--sink",  "qdrant://localhost/col",
    ])
    assert result.exit_code == 1
    # NotImplementedError message must appear somewhere in output
    assert "stub" in result.output or "v0.2" in result.output or "v2.0" in result.output


def test_migrate_unknown_strategy_exits_one():
    result = runner.invoke(app, [
        "migrate",
        "--from", "openai/ada-002", "--to", "openai/3-small",
        "--sink",  "qdrant://localhost/col",
        "--strategy", "magic",
    ])
    assert result.exit_code == 1
    assert "Unknown strategy" in result.output


def test_migrate_shadow_eval_exits_one_with_v2_message():
    result = runner.invoke(app, [
        "migrate",
        "--from", "openai/ada-002", "--to", "openai/3-small",
        "--sink",  "qdrant://localhost/col",
        "--strategy", "shadow-eval",
    ])
    assert result.exit_code == 1
    assert "v2.0" in result.output
