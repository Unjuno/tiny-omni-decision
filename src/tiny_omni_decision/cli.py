from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .io import load_structured_file
from .schema import BaseModelManifest, DatasetManifest

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def validate_model_manifest(path: Path) -> None:
    """Validate a base-model provenance manifest."""
    manifest = BaseModelManifest.model_validate(load_structured_file(path))
    console.print(f"[green]valid[/green] model manifest: {manifest.repo_id}")
    if manifest.revision is None:
        console.print("[yellow]warning:[/yellow] upstream revision is not pinned yet")


@app.command()
def validate_dataset_manifest(path: Path) -> None:
    """Validate a dataset provenance manifest."""
    manifest = DatasetManifest.model_validate(load_structured_file(path))
    console.print(f"[green]valid[/green] dataset manifest: {manifest.dataset_id}")
    if manifest.revision is None:
        console.print("[yellow]warning:[/yellow] dataset revision is not pinned yet")
    if manifest.license.upper() == "UNKNOWN":
        console.print("[yellow]warning:[/yellow] dataset license is unresolved")


@app.command()
def status() -> None:
    """Print the current implementation boundary."""
    console.print("Phase 1 scaffold: ready")
    console.print("Training gate: pin model + dataset revisions and verify ternary runtime support")


if __name__ == "__main__":
    app()
