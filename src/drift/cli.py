"""Drift CLI — drift embed / watch / status / migrate."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="drift",
    help="Spark-native embedding lifecycle. produce → CDC refresh → migrate → audit.",
    add_completion=False,
)


@app.command()
def embed(
    table: str = typer.Option(..., "--table", help="Delta table path or catalog.schema.table"),
    text_col: str = typer.Option(..., "--text-col", help="Column name containing text to embed"),
    model: str = typer.Option(
        "openai/text-embedding-3-small", "--model", help="provider/model-name"
    ),
    sink: str = typer.Option(..., "--sink", help="qdrant://host:port/collection"),
    batch_size: int = typer.Option(128, "--batch-size", help="Rows per API call"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable deduplication"),
    shadow_mode: bool = typer.Option(
        False,
        "--shadow-mode",
        help=(
            "Mock embeddings — no API calls, no cost. "
            "Identical to live mode for dedup/provenance. Safe for CI and local dev."
        ),
    ),
):
    """Embed a Spark table column and upsert vectors to a sink."""
    from .embed import embed as _embed
    from .ledger import Ledger

    mode_tag = " [shadow]" if shadow_mode else ""
    typer.echo(f"[drift embed{mode_tag}] table={table} model={model} sink={sink}")
    try:
        run = _embed(
            df=None,
            source_table=table,
            text_col=text_col,
            model=model,
            sink=sink,
            dedup=not no_dedup,
            batch_size=batch_size,
            shadow_mode=shadow_mode,
            ledger=Ledger(),
        )
        typer.echo(f"  ✓ run_id={run.run_id}")
        typer.echo(
            f"  rows={run.n_rows_processed}  deduped={run.n_rows_deduped}"
            f"  cost=${run.cost_usd:.4f}"
        )
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def watch(
    table: str = typer.Option(..., "--table", help="Delta table to watch"),
    text_col: str = typer.Option(..., "--text-col"),
    sink: str = typer.Option(..., "--sink"),
    model: str = typer.Option("openai/text-embedding-3-small", "--model"),
    since_version: Optional[int] = typer.Option(  # noqa: UP045
        None, "--since-version", help="Delta version to start from"
    ),
    shadow_mode: bool = typer.Option(
        False,
        "--shadow-mode",
        help="Mock embeddings — no API calls, no cost. Same as embed --shadow-mode.",
    ),
):
    """Incrementally refresh embeddings from a Delta table via CDC."""
    from .ledger import Ledger
    from .watch import watch as _watch

    typer.echo(f"[drift watch] table={table} since_version={since_version}")
    try:
        run = _watch(
            source_table=table,
            text_col=text_col,
            sink=sink,
            model=model,
            since_version=since_version,
            shadow_mode=shadow_mode,
            ledger=Ledger(),
        )
        typer.echo(
            f"  ✓ inserted={run.n_inserted}  updated={run.n_updated}"
            f"  deleted={run.n_deleted}"
        )
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def status(
    sink: str = typer.Option(..., "--sink", help="Sink URI to inspect"),
    limit: int = typer.Option(5, "--limit", help="Number of recent runs to show"),
):
    """Show recent embed runs for a sink from the lineage ledger."""
    from .ledger import Ledger

    ledger = Ledger()
    runs = ledger.recent_runs(sink=sink, limit=limit)

    if not runs:
        typer.echo(f"No runs found for sink: {sink}")
        return

    typer.echo(f"Last {len(runs)} run(s) for {sink}:")
    for r in runs:
        typer.echo(
            f"  {r['timestamp'][:19]}  model={r['model']}  "
            f"rows={r['n_rows']}  deduped={r['n_deduped']}  "
            f"cost=${r['cost_usd']:.4f}  duration={r['duration_s']:.1f}s"
        )


@app.command()
def migrate(
    from_model: str = typer.Option(
        ..., "--from", help="Current model string, e.g. openai/text-embedding-ada-002"
    ),
    to_model: str = typer.Option(
        ..., "--to", help="Target model string, e.g. openai/text-embedding-3-small"
    ),
    sink: str = typer.Option(..., "--sink", help="Sink URI of the existing collection"),
    strategy: str = typer.Option(
        "dual-write", "--strategy", help="dual-write | shadow-eval | drift-adapter"
    ),
    shadow_mode: bool = typer.Option(
        False,
        "--shadow-mode",
        help="Mock embeddings — no API calls, no cost. Safe for testing migration flow.",
    ),
):
    """Migrate embeddings to a new model (dual-write re-embeds into <collection>_v2)."""
    from .ledger import Ledger
    from .migrate import STRATEGIES
    from .migrate import migrate as _migrate

    if strategy not in STRATEGIES:
        typer.echo(f"Unknown strategy {strategy!r}. Choose from: {', '.join(STRATEGIES)}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"[drift migrate] {from_model} → {to_model}  strategy={strategy}  sink={sink}")

    try:
        run = _migrate(
            from_model=from_model,
            to_model=to_model,
            sink=sink,
            strategy=strategy,
            shadow_mode=shadow_mode,
            ledger=Ledger(),
        )
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1) from None

    if run.adapter_path:
        # drift-adapter: no new collection, old index stays unchanged
        typer.echo("  ✓ Adapter fitted and saved")
        typer.echo(f"  ARR:     {run.arr:.3f}  (threshold: 0.97)")
        typer.echo(f"  Adapter: {run.adapter_path}")
        typer.echo(f"  Trained on {run.n_source} paired samples from {sink}")
        typer.echo("\n  Apply at query time:")
        typer.echo("    from drift import DriftAdapter")
        typer.echo(f"    adapter = DriftAdapter.load('{run.adapter_path}')")
        typer.echo("    adapted_vec = adapter.predict(new_model_query_vec)")
        typer.echo(f"    # search {sink} with adapted_vec (no reindex needed)")
        typer.echo("\n  Next steps:")
        typer.echo("    1. Catch-up: drift watch --table <source-table> --text-col <col> \\")
        typer.echo(f"                       --sink {sink} --model {to_model}")
        typer.echo("       (syncs docs added after the adapter was trained)")
        typer.echo(f"    2. Monitor:  drift status --sink {sink}")
    else:
        # dual-write: new collection created
        icon = "✓" if run.n_migrated == run.n_source else "⚠"
        typer.echo(f"  {icon} Migration complete: {run.n_migrated}/{run.n_source} vectors written")
        typer.echo(f"  New collection: {run.sink_v2}")
        typer.echo(f"  Duration: {run.duration_s:.1f}s")

        if run.n_migrated != run.n_source:
            missing = run.n_source - run.n_migrated
            typer.echo(f"\n  ⚠ {missing} vectors missing from new collection.", err=True)
            typer.echo(
                "  Likely cause: those points have no source_text payload "
                "(embedded by a tool other than drift embed).",
                err=True,
            )

        typer.echo("\n  Next steps:")
        typer.echo("    1. Catch-up:  drift watch --table <source-table> --text-col <col> \\")
        typer.echo(f"                        --sink {run.sink_v2} --model {to_model}")
        typer.echo("       (syncs docs added during migration — run once before validating)")
        typer.echo(f"    2. Validate:  run your real queries against {run.sink_v2}")
        typer.echo(f"    3. Cutover:   update your app to query {run.sink_v2}")
        typer.echo(f"    4. Monitor:   drift status --sink {run.sink_v2}")
        typer.echo(f"    5. Cleanup:   delete {run.sink} when satisfied")


if __name__ == "__main__":
    app()
