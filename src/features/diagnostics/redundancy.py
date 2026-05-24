# src/features/diagnostics/redundancy.py
"""
Diagnósticos de redundancia entre features EEG-HMM.

Objetivo:
Detectar features altamente correlacionadas o geométricamente redundantes
antes del PCA/HMM.

Esto NO implica que una feature sea "mala".
La redundancia puede ser:
    - esperada (ej: alpha_envelope ~ RMS)
    - útil (PCA la comprime)
    - peligrosa (domina componentes)

Incluye:

1. Correlation matrix
2. High-correlation pairs
3. Mutual redundancy score
4. Effective rank approximation
5. Feature dominance diagnostics
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from itertools import combinations


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class RedundantPair:
    feature_a: str
    feature_b: str

    correlation: float

    interpretation: str


@dataclass
class RedundancyReport:
    n_features: int

    mean_abs_correlation: float
    max_abs_correlation: float

    effective_rank: float

    redundant_pairs: list[RedundantPair]

    dominance_ratio: float

    interpretation: str


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlación segura.
    """
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def _flatten_feature(X: np.ndarray) -> np.ndarray:
    """
    Convierte una feature multicanal en vector 1D.

    Input:
        shape = (n_windows, n_dims)

    Output:
        shape = (n_windows * n_dims,)
    """
    return X.reshape(-1)


# ---------------------------------------------------------------------
# Pairwise redundancy
# ---------------------------------------------------------------------

def compute_pairwise_redundancy(
    feature_data: dict[str, np.ndarray],
    high_corr_threshold: float = 0.85,
) -> list[RedundantPair]:
    """
    Detecta pares redundantes.

    Args:
        feature_data:
            {
                "alpha_envelope": array(n_windows, n_dims),
                "hjorth_mobility": ...
            }

    Returns:
        Lista de pares altamente correlacionados.
    """

    pairs = []

    names = list(feature_data.keys())

    for a, b in combinations(names, 2):

        xa = _flatten_feature(feature_data[a])
        xb = _flatten_feature(feature_data[b])

        n = min(len(xa), len(xb))

        xa = xa[:n]
        xb = xb[:n]

        corr = _safe_corr(xa, xb)

        abs_corr = abs(corr)

        if abs_corr >= high_corr_threshold:

            if abs_corr >= 0.95:
                interp = "extreme_redundancy"

            elif abs_corr >= 0.90:
                interp = "high_redundancy"

            else:
                interp = "moderate_redundancy"

            pairs.append(
                RedundantPair(
                    feature_a=a,
                    feature_b=b,
                    correlation=float(corr),
                    interpretation=interp,
                )
            )

    return sorted(
        pairs,
        key=lambda p: abs(p.correlation),
        reverse=True,
    )


# ---------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------

def compute_feature_correlation_matrix(
    feature_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Construye matriz feature-feature.

    Returns:
        corr_matrix, ordered_names
    """

    names = list(feature_data.keys())

    flattened = []

    for name in names:
        flattened.append(_flatten_feature(feature_data[name]))

    # Igualar longitudes
    min_len = min(len(x) for x in flattened)

    flattened = [x[:min_len] for x in flattened]

    X = np.vstack(flattened)

    corr_matrix = np.corrcoef(X)

    return corr_matrix, names


# ---------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------

def compute_effective_rank(
    corr_matrix: np.ndarray,
) -> float:
    """
    Effective rank basado en entropy rank.

    Referencia:
        Roy & Vetterli (2007)

    Intuición:
        - rank ~ n_features → features diversas
        - rank << n_features → alta redundancia
    """

    eigvals = np.linalg.eigvalsh(corr_matrix)

    eigvals = np.clip(eigvals, 1e-12, None)

    p = eigvals / np.sum(eigvals)

    entropy = -np.sum(p * np.log(p))

    effective_rank = np.exp(entropy)

    return float(effective_rank)


# ---------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------

def compute_dominance_ratio(
    corr_matrix: np.ndarray,
) -> float:
    """
    Dominancia geométrica.

    Ratio:
        λ1 / Σλ

    Intuición:
        Si una sola dirección explica demasiado,
        el espacio está dominado por una feature/familia.
    """

    eigvals = np.linalg.eigvalsh(corr_matrix)

    eigvals = np.sort(eigvals)[::-1]

    ratio = eigvals[0] / np.sum(eigvals)

    return float(ratio)


# ---------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------

def analyze_redundancy(
    feature_data: dict[str, np.ndarray],
    high_corr_threshold: float = 0.85,
) -> RedundancyReport:
    """
    Diagnóstico completo de redundancia.
    """

    corr_matrix, names = compute_feature_correlation_matrix(
        feature_data
    )

    # -------------------------------------------------------------
    # Estadísticas globales
    # -------------------------------------------------------------
    upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    abs_upper = np.abs(upper)

    mean_abs_corr = float(np.mean(abs_upper))
    max_abs_corr = float(np.max(abs_upper))

    # -------------------------------------------------------------
    # Effective rank
    # -------------------------------------------------------------
    eff_rank = compute_effective_rank(corr_matrix)

    # -------------------------------------------------------------
    # Dominancia
    # -------------------------------------------------------------
    dominance_ratio = compute_dominance_ratio(corr_matrix)

    # -------------------------------------------------------------
    # Pares redundantes
    # -------------------------------------------------------------
    redundant_pairs = compute_pairwise_redundancy(
        feature_data,
        high_corr_threshold=high_corr_threshold,
    )

    # -------------------------------------------------------------
    # Interpretación global
    # -------------------------------------------------------------
    n_features = len(names)

    rank_ratio = eff_rank / max(n_features, 1)

    if rank_ratio > 0.85:
        interpretation = "low_redundancy"

    elif rank_ratio > 0.60:
        interpretation = "moderate_redundancy"

    else:
        interpretation = "high_redundancy"

    return RedundancyReport(
        n_features=n_features,

        mean_abs_correlation=mean_abs_corr,
        max_abs_correlation=max_abs_corr,

        effective_rank=eff_rank,

        redundant_pairs=redundant_pairs,

        dominance_ratio=dominance_ratio,

        interpretation=interpretation,
    )

# ---------------------------------------------------------------------
# PUBLIC API WRAPPER
# ---------------------------------------------------------------------

def compute_redundancy_diagnostics(
    X_global: np.ndarray,
    feature_mapping: list[dict],
) -> dict:
    """
    Wrapper compatible con el pipeline de validación.

    Parameters
    ----------
    X_global :
        shape = (n_windows, n_total_features)

    feature_mapping :
        Lista producida por:
        reconstruct_feature_column_mapping()

    Returns
    -------
    dict serializable JSON
    """

    # -------------------------------------------------------------
    # Reconstruir dict feature -> matriz
    # -------------------------------------------------------------
    feature_data = {}

    for item in feature_mapping:

        name = item["feature_name"]
        cols = item["columns"]

        feature_data[name] = X_global[:, cols]

    # -------------------------------------------------------------
    # Ejecutar análisis
    # -------------------------------------------------------------
    report = analyze_redundancy(feature_data)

    # -------------------------------------------------------------
    # Serializar dataclasses
    # -------------------------------------------------------------
    redundant_pairs = []

    for pair in report.redundant_pairs:

        redundant_pairs.append(
            {
                "feature_a": pair.feature_a,
                "feature_b": pair.feature_b,
                "correlation": float(pair.correlation),
                "interpretation": pair.interpretation,
            }
        )

    return {
        "n_features": int(report.n_features),

        "mean_abs_correlation":
            float(report.mean_abs_correlation),

        "max_abs_correlation":
            float(report.max_abs_correlation),

        "effective_rank":
            float(report.effective_rank),

        "dominance_ratio":
            float(report.dominance_ratio),

        "interpretation":
            report.interpretation,

        "redundant_pairs":
            redundant_pairs,
    }