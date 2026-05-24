# src/utils/paths.py
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def clean_path(yaml_path: str) -> str:
    """
    Resuelve rutas del YAML (que usan ../../ relativos al config)
    a rutas absolutas desde la raíz del proyecto.
    """
    normalized = yaml_path.replace('../../', '').replace('../', '')
    return os.path.normpath(PROJECT_ROOT / normalized)
