# orchestration/registry.py
"""
Registro central de etapas del pipeline.
Soporta múltiples tipos de pipeline según pipeline_type en el YAML del experimento.

Tipos disponibles:
  - "feature"  (default): preprocess → extract → pca → hmm → decode
  - "tde":                 preprocess → embed   → hmm → decode
"""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Pipeline Featured-HMM (original) ────────────────────────────────────────
FEATURE_PIPELINE_STAGES = {
    "preprocess": {
        "script": PROJECT_ROOT / "scripts" / "preprocessing" / "preprocess_task_batch.py",
        "label": "Preprocesamiento (raw → .fif)",
    },
    "extract": {
        "script": PROJECT_ROOT / "scripts" / "features" / "01_extract_features.py",
        "label": "Extracción de features manuales",
    },
    "pca": {
        "script": PROJECT_ROOT / "scripts" / "features" / "02_fit_pca.py",
        "label": "Ajuste de PCA",
    },
    "hmm": {
        "script": PROJECT_ROOT / "scripts" / "training" / "03_train_hmm.py",
        "label": "Entrenamiento HMM",
    },
    "decode": {
        "script": PROJECT_ROOT / "scripts" / "decoding" / "decode_feature_hmm.py",
        "label": "Decoding (FO, Dwell Time, Transition Rate)",
    },
}

# ── Pipeline TDE-HMM ─────────────────────────────────────────────────────────
TDE_PIPELINE_STAGES = {
    "preprocess": {
        "script": PROJECT_ROOT / "scripts" / "preprocessing" / "preprocess_task_batch.py",
        "label": "Preprocesamiento (raw → .fif)",
    },
    "embed": {
        "script": PROJECT_ROOT / "scripts" / "features" / "compute_tde_embeddings.py",
        "label": "TDE Embedding + PCA (señal raw con lags temporales)",
    },
    "hmm": {
        "script": PROJECT_ROOT / "scripts" / "training" / "train_tde_hmm.py",
        "label": "Entrenamiento HMM sobre espacio TDE",
    },
    "decode": {
        "script": PROJECT_ROOT / "scripts" / "decoding" / "decode_tde_hmm.py",
        "label": "Decoding TDE (FO, Dwell Time, Transition Rate, topomapas)",
    },
}

# ── Mapa de tipos de pipeline ────────────────────────────────────────────────
PIPELINE_REGISTRY = {
    "feature": FEATURE_PIPELINE_STAGES,
    "tde":     TDE_PIPELINE_STAGES,
}

# ── Compatibilidad con código existente (default = feature) ──────────────────
PIPELINE_STAGES = FEATURE_PIPELINE_STAGES
STAGE_NAMES     = list(FEATURE_PIPELINE_STAGES.keys())


def get_pipeline_type(config_path: Path) -> str:
    """Lee pipeline_type del YAML del experimento. Default: 'feature'."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("experiment", {}).get("pipeline_type", "feature")


def get_stages_for_config(config_path: Path) -> dict:
    """Devuelve el dict de etapas correspondiente al pipeline_type del YAML."""
    pipeline_type = get_pipeline_type(config_path)
    if pipeline_type not in PIPELINE_REGISTRY:
        valid = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(
            f"pipeline_type='{pipeline_type}' no reconocido. "
            f"Válidos: {valid}"
        )
    return PIPELINE_REGISTRY[pipeline_type]


def get_stages_from(start_stage: str, config_path: Path | None = None) -> list[dict]:
    """
    Devuelve las etapas desde start_stage en adelante.
    Si config_path es None, usa el pipeline 'feature' por compatibilidad.
    """
    if config_path is not None:
        stages_dict = get_stages_for_config(config_path)
    else:
        stages_dict = FEATURE_PIPELINE_STAGES

    stage_names = list(stages_dict.keys())
    if start_stage not in stage_names:
        valid = ", ".join(stage_names)
        raise ValueError(
            f"Etapa desconocida: '{start_stage}'. "
            f"Etapas válidas para este pipeline: {valid}"
        )

    stages = []
    found = False
    for name, info in stages_dict.items():
        if name == start_stage:
            found = True
        if found:
            stages.append({"name": name, **info})
    return stages


def get_all_stages(config_path: Path | None = None) -> list[dict]:
    """Devuelve todas las etapas del pipeline correspondiente."""
    if config_path is not None:
        stages_dict = get_stages_for_config(config_path)
    else:
        stages_dict = FEATURE_PIPELINE_STAGES
    return [{"name": name, **info} for name, info in stages_dict.items()]