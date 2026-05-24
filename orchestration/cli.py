import sys
from pathlib import Path
from typing import Optional
import typer

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from orchestration.registry import PIPELINE_STAGES
from orchestration.runner import run_pipeline

app = typer.Typer(name="hmm", add_completion=False, no_args_is_help=True)


@app.command()
def run(
    config: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True,
    ),
    from_stage: Optional[str] = typer.Option(None, "--from"),
):
    """Ejecuta el pipeline completo o desde una etapa especifica."""
    if from_stage is not None:
        from orchestration.registry import get_stages_for_config
        valid_stages = list(get_stages_for_config(config).keys())
        if from_stage not in valid_stages:
            valid = ", ".join(valid_stages)
            typer.echo(typer.style(
                f"Error: Etapa '{from_stage}' no reconocida. Validas: {valid}",
                fg=typer.colors.RED, bold=True,
            ))
            raise typer.Exit(code=1)
    success = run_pipeline(config_path=config, from_stage=from_stage)
    raise typer.Exit(code=0 if success else 1)


@app.command()
def stages(
    config: Optional[Path] = typer.Argument(None),
):
    """Lista las etapas del pipeline."""
    if config is not None and Path(config).exists():
        from orchestration.registry import get_stages_for_config, get_pipeline_type
        pipeline_type = get_pipeline_type(Path(config))
        stages_dict = get_stages_for_config(Path(config))
        typer.echo(f"\nEtapas para pipeline '{pipeline_type}':\n")
    else:
        stages_dict = PIPELINE_STAGES
        typer.echo("\nEtapas Featured-HMM (default):\n")
    for i, (name, info) in enumerate(stages_dict.items(), start=1):
        icon = "OK" if info["script"].exists() else "MISSING"
        typer.echo(f"  {i}. [{name}] {info['label']} [{icon}]")
    typer.echo()


@app.command()
def inspect(
    config: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True,
    ),
):
    """Muestra el run.json del ultimo run del experimento."""
    import json
    from orchestration.runner import _resolve_output_dir
    output_dir = _resolve_output_dir(config)
    run_json_path = output_dir / "run.json"
    if not run_json_path.exists():
        typer.echo("No se encontro run.json")
        raise typer.Exit(1)
    data = json.loads(run_json_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    app()


if __name__ == "__main__":
    main()
