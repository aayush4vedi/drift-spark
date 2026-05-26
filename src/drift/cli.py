"""Drift CLI — drift embed / watch / status / migrate."""

import typer
from typing import Optional

app = typer.Typer(
    name="drift",
    help="Spark-native embedding lifecycle. produce → CDC refresh → migrate → audit.",
    add_completion=False,
)


@app.command()
def embed(
    table: str = typer.Option(..., "--table", help="Delta table path or catalog.schema.table"),
    text_col: str = typer.Option(..., "--text-col", help="Column to embed"),
    model: str = typer.Option("openai/text-embedding-3-small", "--model"),
    sink: str = typer.Option(..., "--sink", help="qdrant://host:port/collection or pg://..."),
    batch_size: int = typer.Option(128, "--batch-size"),
):
    """Embed a Spark table column and upsert vectors to a sink."""
    typer.echo(f"[drift embed] table={table} model={model} sink={sink}")
    raise typer.Exit(code=1)  # stub — implementation in next iteration


@app.command()
def watch(
    table: str = typer.Option(..., "--table"),
    text_col: str = typer.Option(..., "--text-col"),
    sink: str = typer.Option(..., "--sink"),
    model: str = typer.Option("openai/text-embedding-3-small", "--model"),
    since_version: Optional[int] = typer.Option(None, "--since-version"),
):
    """Incrementally refresh embeddings from a Delta table via CDC."""
    typer.echo(f"[drift watch] table={table} since_version={since_version}")
    raise typer.Exit(code=1)  # stub — implementation next iteration


@app.command()
def status(
    sink: str = typer.Option(..., "--sink", help="Sink URI to inspect"),
):
    """Show last 5 embed runs for a sink from the lineage ledger."""
    from .ledger import Ledger
    ledger = Ledger()
    typer.echo(f"[drift status] sink={sink}")
    for row in ledger.cost_by_model():
        typer.echo(f"  model={row['model']}  cost=${row['cost_usd']:.4f}")


@app.command()
def migrate(
    from_model: str = typer.Option(..., "--from"),
    to_model: str = typer.Option(..., "--to"),
    sink: str = typer.Option(..., "--sink"),
    strategy: str = typer.Option("dual-write", "--strategy"),
):
    """Migrate embeddings between models (v1.0: dual-write only)."""
    if strategy != "dual-write":
        typer.echo(f"strategy={strategy!r} coming in v2.0. Use --strategy dual-write.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[drift migrate] from={from_model} to={to_model} strategy={strategy}")
    raise typer.Exit(code=1)  # stub — implementation next iteration


if __name__ == "__main__":
    app()