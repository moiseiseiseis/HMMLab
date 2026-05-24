# orchestration/runner.py
"""
Runner del pipeline: ejecuta cada etapa vía subprocess.run.
Soporta múltiples tipos de pipeline (feature, tde) según pipeline_type en el YAML.
"""

import subprocess
import sys
import yaml
import datetime
from pathlib import Path

from orchestration.registry import (
    PROJECT_ROOT,
    get_all_stages,
    get_stages_from,
    get_pipeline_type,
)
from orchestration.tracker import RunTracker


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)

def _separator(char: str = "-", width: int = 60) -> None:
    print(char * width, flush=True)


def _resolve_output_dir(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_output_dir: str = cfg["paths"]["output_dir"]
    clean = raw_output_dir.replace("../../", "").replace("../", "")
    base_dir = (PROJECT_ROOT / clean).resolve()
    experiment_name: str = cfg["experiment"]["name"]
    return base_dir / experiment_name


def _get_experiment_name(config_path: Path) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["experiment"]["name"]


def _run_stage(stage: dict, config_path: Path) -> tuple[int, str | None]:
    script_path: Path = stage["script"]
    if not script_path.exists():
        return 1, f"Script no encontrado: {script_path}"
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    _log(f"->  Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=None, stderr=None, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        return result.returncode, (
            f"El script '{script_path.name}' terminó con código {result.returncode}. "
            f"Revisa el output de arriba para detalles."
        )
    return 0, None


def run_pipeline(config_path: Path, from_stage: str | None = None) -> bool:
    config_path = config_path.resolve()

    # Detecta pipeline_type desde el YAML
    pipeline_type   = get_pipeline_type(config_path)
    all_stages      = get_all_stages(config_path)       # ← usa config_path
    all_stage_names = [s["name"] for s in all_stages]

    # Valida from_stage contra el pipeline correcto
    if from_stage is not None:
        if from_stage not in all_stage_names:
            valid = ", ".join(all_stage_names)
            _log(f"ERROR  ERROR: Etapa desconocida '{from_stage}'. Válidas: {valid}")
            return False

    try:
        output_dir      = _resolve_output_dir(config_path)
        experiment_name = _get_experiment_name(config_path)
    except (FileNotFoundError, KeyError) as e:
        _log(f"ERROR  ERROR al leer el config: {e}")
        return False

    if from_stage is not None:
        active_stages = get_stages_from(from_stage, config_path)  # ← usa config_path
        skipped_names = [
            s["name"] for s in all_stages
            if s["name"] not in [a["name"] for a in active_stages]
        ]
    else:
        active_stages = all_stages
        skipped_names = []

    tracker = RunTracker(
        output_dir=output_dir,
        config_path=config_path,
        experiment_name=experiment_name,
        stage_names=all_stage_names,
    )
    run_id = tracker.start_run(skipped_stages=skipped_names)

    _separator("=")
    _log(f">>>  INICIANDO PIPELINE HMM [{pipeline_type.upper()}]")
    _log(f"    Experimento  : {experiment_name}")
    _log(f"    Pipeline     : {pipeline_type}")
    _log(f"    Config       : {config_path}")
    _log(f"    Run ID       : {run_id}")
    _log(f"    Output dir   : {output_dir}")
    if from_stage:
        _log(f"    Reanudando desde: '{from_stage}'")
        _log(f"    Etapas saltadas : {skipped_names}")
    _log(f"    Etapas activas: {[s['name'] for s in active_stages]}")
    _separator("=")

    pipeline_success = True

    for i, stage in enumerate(active_stages, start=1):
        name  = stage["name"]
        label = stage["label"]
        total = len(active_stages)

        _separator()
        _log(f"[{i}/{total}] ETAPA: {label.upper()} ('{name}')")
        _separator()

        tracker.start_stage(name)
        returncode, error = _run_stage(stage, config_path)
        tracker.finish_stage(name, returncode=returncode, error=error)

        if returncode != 0:
            _separator("=")
            _log(f"ERROR  PIPELINE DETENIDO en etapa '{name}'")
            _log(f"    {error}")
            _log(f"    Para reanudar: hmm run {config_path} --from {name}")
            _log(f"    run.json en  : {tracker.run_json_location}")
            _separator("=")
            pipeline_success = False
            break

        _log(f"OK  Etapa '{name}' completada.")

    tracker.finish_run(success=pipeline_success)

    _separator("=")
    if pipeline_success:
        _log(f"DONE  PIPELINE COMPLETADO EXITOSAMENTE")
    else:
        _log(f"FAIL  PIPELINE FALLIDO")
    _log(f"    run.json: {tracker.run_json_location}")
    _separator("=")

    return pipeline_success