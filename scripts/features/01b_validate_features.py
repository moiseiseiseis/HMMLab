# scripts/features/01b_validate_features.py
"""
Validación científica de features EEG-HMM.

Fase 2 del pipeline:
    - Diagnósticos estadísticos
    - Diagnósticos temporales
    - Diagnósticos inter-sujeto
    - Redundancia
    - Scoring heurístico HMM

Este script:
    1. Carga features ya extraídas (.npy)
    2. Reconstruye matriz global
    3. Reconstruye metadata por columna
    4. Ejecuta diagnósticos modulares
    5. Guarda reportes JSON

NO entrena modelos.
NO hace PCA.
NO modifica los datos.

Uso:
    python scripts/features/01b_validate_features.py \
        --config configs/experiments/mi_exp.yaml
"""

from __future__ import annotations

import os
import sys
import glob
import json
import yaml
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm




sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT, clean_path


# ---------------------------------------------------------------------
# IMPORT FEATURES (REGISTRA TODO)
# ---------------------------------------------------------------------

import src.features

from src.features.registry import REGISTRY

# ---------------------------------------------------------------------
# IMPORT DIAGNOSTICS
# ---------------------------------------------------------------------

from src.features.diagnostics.distribution import (
    compute_distribution_diagnostics,
)

from src.features.diagnostics.temporality import (
    compute_temporal_diagnostics,
)

from src.features.diagnostics.redundancy import (
    compute_redundancy_diagnostics,
)

from src.features.diagnostics.inter_subject import (
    compute_inter_subject_diagnostics,
)

from src.features.diagnostics.hmm_score import (
    compute_hmm_suitability_score,
)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def load_feature_matrix(features_dir: str):
    """
    Carga todas las matrices de features.

    Returns
    -------
    X_global : np.ndarray
        Shape (n_total_windows, n_features)

    subject_ids : list[str]

    subject_slices : dict
        subject_id -> slice(start, end)
    """

    feature_files = sorted(
        glob.glob(os.path.join(features_dir, "*_features.npy"))
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No se encontraron archivos *_features.npy en:\n{features_dir}"
        )

    matrices = []
    subject_ids = []
    subject_slices = {}

    cursor = 0

    print("\nCargando matrices de features...\n")

    for fp in tqdm(feature_files):

        subject_id = os.path.basename(fp).replace("_features.npy", "")

        X = np.load(fp)

        start = cursor
        end = cursor + X.shape[0]

        subject_slices[subject_id] = slice(start, end)

        cursor = end

        matrices.append(X)
        subject_ids.append(subject_id)

    X_global = np.vstack(matrices)

    return X_global, subject_ids, subject_slices


def reconstruct_feature_column_mapping(
    feature_objects,
    n_channels: int,
):
    """
    Reconstruye qué columnas pertenecen a qué feature.

    Ejemplo:
        hjorth_mobility -> cols [0..18]
        hjorth_complexity -> cols [19..37]

    Returns
    -------
    list[dict]
    """

    mapping = []

    cursor = 0

    for feat in feature_objects:

        dim = feat.output_dim(n_channels)

        cols = list(range(cursor, cursor + dim))

        mapping.append(
            {
                "feature_name": feat.metadata.name,
                "display_name": feat.metadata.display_name,
                "metadata": feat.metadata,
                "columns": cols,
            }
        )

        cursor += dim

    return mapping


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def serialize_metadata(meta):
    return meta.to_dict()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description="Validador científico de features EEG-HMM"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al YAML del experimento",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # LOAD CONFIG
    # -----------------------------------------------------------------

    with open(args.config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    exp_name = exp_cfg["experiment"]["name"]

    feature_flags = exp_cfg.get("features", {})

    features_dir = clean_path(
        exp_cfg["paths"]["features_dir"],
    )

    output_root = clean_path(
        exp_cfg["paths"]["output_dir"],
    )

    diagnostics_dir = os.path.join(
        output_root,
        exp_name,
        "diagnostics",
    )

    ensure_output_dir(diagnostics_dir)

    # -----------------------------------------------------------------
    # RESOLVE FEATURES
    # -----------------------------------------------------------------

    feature_objects = REGISTRY.resolve_from_yaml_flags(
        feature_flags
    )

    if not feature_objects:
        raise RuntimeError(
            "No hay features activas en el YAML."
        )

    print("\n============================================================")
    print(" VALIDACIÓN CIENTÍFICA DE FEATURES")
    print("============================================================")

    print(f"Experimento : {exp_name}")
    print(f"Features dir: {features_dir}")
    print(f"Output dir  : {diagnostics_dir}")

    print("\nFeatures activas:")

    for feat in feature_objects:
        print(
            f"  - {feat.metadata.name:<30}"
            f"[{feat.metadata.category}]"
        )

    # -----------------------------------------------------------------
    # LOAD MATRICES
    # -----------------------------------------------------------------

    X_global, subject_ids, subject_slices = load_feature_matrix(
        features_dir
    )

    print("\n============================================================")
    print(" MATRIZ GLOBAL")
    print("============================================================")

    print(f"Shape global: {X_global.shape}")
    print(f"Sujetos     : {len(subject_ids)}")

    # -----------------------------------------------------------------
    # RECONSTRUCT COLUMN MAPPING
    # -----------------------------------------------------------------

    n_total_features = X_global.shape[1]

    n_channels = None

    for feat in feature_objects:
        if feat.metadata.n_channels_dependent:
            # Inferencia simple:
            # asumimos que TODAS las channel-dependent
            # usan mismo número de canales
            #
            # total_cols / num_features
            pass

    # Inferencia robusta:
    #
    # Si todas son channel-dependent:
    #
    # total_cols = n_features * n_channels
    #
    # entonces:
    #
    # n_channels = total_cols / sum(dim_per_channel)
    #

    dims_per_feature = []

    for feat in feature_objects:
        if feat.metadata.n_channels_dependent:
            dims_per_feature.append(1)
        else:
            dims_per_feature.append(0)

    n_channel_features = sum(dims_per_feature)

    if n_channel_features == 0:
        n_channels = 1
    else:
        n_channels = n_total_features // n_channel_features

    feature_mapping = reconstruct_feature_column_mapping(
        feature_objects,
        n_channels=n_channels,
    )

    print(f"\nCanales inferidos: {n_channels}")

    # -----------------------------------------------------------------
    # RUN DIAGNOSTICS
    # -----------------------------------------------------------------

    full_report = defaultdict(dict)

    print("\n============================================================")
    print(" DIAGNÓSTICOS")
    print("============================================================")

    for item in feature_mapping:

        feature_name = item["feature_name"]

        metadata = item["metadata"]

        cols = item["columns"]

        X_feat = X_global[:, cols]

        print(f"\n→ {feature_name}")
        print(f"  shape: {X_feat.shape}")

        # -------------------------------------------------------------
        # DISTRIBUTION
        # -------------------------------------------------------------

        dist_report = compute_distribution_diagnostics(
            X_feat,
            metadata=metadata,
        )

        # -------------------------------------------------------------
        # TEMPORALITY
        # -------------------------------------------------------------

        temp_report = compute_temporal_diagnostics(
            X_feat,
            subject_slices=subject_slices,
        )

        # -------------------------------------------------------------
        # INTER SUBJECT
        # -------------------------------------------------------------

        inter_report = compute_inter_subject_diagnostics(
            X_feat,
            subject_slices=subject_slices,
        )

        # -------------------------------------------------------------
        # HMM SCORE
        # -------------------------------------------------------------

        hmm_score = compute_hmm_suitability_score(
            distribution_report=dist_report,
            temporal_report=temp_report,
            inter_subject_report=inter_report,
            metadata=metadata,
        )

        full_report[feature_name] = {
            "metadata": serialize_metadata(metadata),
            "distribution": dist_report,
            "temporality": temp_report,
            "inter_subject": inter_report,
            "hmm_score": hmm_score,
        }

    # -----------------------------------------------------------------
    # REDUNDANCY
    # -----------------------------------------------------------------

    print("\n============================================================")
    print(" REDUNDANCIA")
    print("============================================================")

    redundancy_report = compute_redundancy_diagnostics(
        X_global,
        feature_mapping=feature_mapping,
    )

    full_report["_global"] = {
        "redundancy": redundancy_report,
    }

    # -----------------------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------------------

    out_json = os.path.join(
        diagnostics_dir,
        "feature_validation_report.json",
    )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            full_report,
            f,
            indent=2,
        )

    print("\n============================================================")
    print(" VALIDACIÓN COMPLETADA")
    print("============================================================")

    print(f"\nReporte guardado en:\n{out_json}")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()