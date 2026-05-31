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
    model: str = typer.Option("openai/text-embedding-3-small", "--model", help="provider/model-name"),
    sink: str = typer.Option(..., "--sink", help="qdrant://host:port/collection"),
    batch_size: int = typer.Option(128, "--batch-size", help="Rows per API call"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable deduplication"),
    shadow_mode: bool = typer.Option(
        False, "--shadow-mode",
        help="Mock embeddings — no API calls, no cost. Identical to live mode for dedup/provenance. Safe for CI and local dev.",
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
        typer.echo(f"  rows={run.n_rows_processed}  deduped={run.n_rows_deduped}  cost=${run.cost_usd:.4f}")
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def watch(
    table: str = typer.Option(..., "--table", help="Delta table to watch"),
    text_col: str = typer.Option(..., "--text-col"),
    sink: str = typer.Option(..., "--sink"),
    model: str = typer.Option("openai/text-embedding-3-small", "--model"),
    since_version: Optional[int] = typer.Option(None, "--since-version", help="Delta version to start from"),
    shadow_mode: bool = typer.Option(
        False, "--shadow-mode",
        help="Mock embeddings — no API calls, no cost. Same as embed --shadow-mode.",
    ),
):
    """Incrementally refresh embeddings from a Delta table via CDC."""
    from .watch import watch as _watch
    from .ledger import Ledger

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
        typer.echo(f"  ✓ inserted={run.n_inserted}  updated={run.n_updated}  deleted={run.n_deleted}")
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1)


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
    from_model: str = typer.Option(..., "--from", help="Current model string"),
    to_model: str = typer.Option(..., "--to", help="Target model string"),
    sink: str = typer.Option(..., "--sink", help="Sink URI of the existing collection"),
    strategy: str = typer.Option("dual-write", "--strategy", help="dual-write | shadow-eval | drift-adapter"),
):
    """Migrate embeddings between models (v1.0: dual-write only)."""
    from .migrate import migrate as _migrate, STRATEGIES
    from .ledger import Ledger

    if strategy not in STRATEGIES:
        typer.echo(f"Unknown strategy {strategy!r}. Choose from: {', '.join(STRATEGIES)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[drift migrate] {from_model} → {to_model}  strategy={strategy}  sink={sink}")
    try:
        run = _migrate(from_model=from_model, to_model=to_model, sink=sink,
                       strategy=strategy, ledger=Ledger())
        typer.echo(f"  ✓ migrated={run.n_migrated}  new_collection={run.collection_new}")
    except NotImplementedError as e:
        typer.echo(f"  [stub] {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
