# src/features/diagnostics/distribution.py
"""
Diagnósticos de distribución para features EEG-HMM.

Objetivo:
---------
Evaluar si una feature es compatible con los supuestos
de un Gaussian HMM.

Se analizan:
    - media
    - desviación estándar
    - skewness
    - kurtosis
    - normalidad aproximada
    - outliers
    - rango dinámico
    - gaussianity heuristic score

IMPORTANTE:
------------
Esto NO reemplaza inspección visual.
Es un diagnóstico heurístico automatizado.

API:
----
compute_distribution_diagnostics(
    X_feat,
    metadata,
) -> dict
"""

from __future__ import annotations

import numpy as np

from scipy.stats import (
    skew,
    kurtosis,
    normaltest,
)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def _safe_float(x):
    """
    Convierte a float serializable JSON.
    """

    if x is None:
        return None

    if np.isnan(x):
        return None

    if np.isinf(x):
        return None

    return float(x)


def _flatten_feature(X_feat: np.ndarray) -> np.ndarray:
    """
    Convierte shape (n_windows, n_dims)
    a vector 1D para análisis global.
    """

    return np.asarray(X_feat).ravel()


def _compute_outlier_fraction(
    x: np.ndarray,
    z_thresh: float = 3.0,
) -> float:
    """
    Fracción de valores extremos.
    """

    std = np.std(x)

    if std < 1e-12:
        return 0.0

    z = (x - np.mean(x)) / std

    frac = np.mean(np.abs(z) > z_thresh)

    return float(frac)


def _gaussianity_score(
    skewness_abs: float,
    kurtosis_abs: float,
    outlier_fraction: float,
    normality_p: float | None,
) -> float:
    """
    Heurística [0, 1].

    1.0 → muy gaussiana
    0.0 → extremadamente no gaussiana

    No pretende ser una métrica formal.
    """

    score = 1.0

    # -------------------------------------------------------------
    # Penalización por skewness
    # -------------------------------------------------------------

    score -= min(skewness_abs / 5.0, 0.35)

    # -------------------------------------------------------------
    # Penalización por kurtosis
    # -------------------------------------------------------------

    score -= min(kurtosis_abs / 10.0, 0.30)

    # -------------------------------------------------------------
    # Penalización por outliers
    # -------------------------------------------------------------

    score -= min(outlier_fraction * 5.0, 0.20)

    # -------------------------------------------------------------
    # Penalización por normalidad
    # -------------------------------------------------------------

    if normality_p is not None:

        if normality_p < 1e-6:
            score -= 0.15

        elif normality_p < 1e-3:
            score -= 0.10

        elif normality_p < 0.05:
            score -= 0.05

    return float(np.clip(score, 0.0, 1.0))


def _distribution_label(score: float) -> str:
    """
    Traduce score a categoría.
    """

    if score >= 0.80:
        return "excellent"

    if score >= 0.65:
        return "good"

    if score >= 0.45:
        return "acceptable"

    if score >= 0.25:
        return "poor"

    return "very_poor"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def compute_distribution_diagnostics(
    X_feat: np.ndarray,
    metadata=None,
) -> dict:
    """
    Ejecuta diagnósticos de distribución.

    Parameters
    ----------
    X_feat :
        Shape (n_windows, n_feature_dims)

    metadata :
        FeatureMetadata opcional.

    Returns
    -------
    dict
        Reporte serializable JSON.
    """

    # -----------------------------------------------------------------
    # FLATTEN
    # -----------------------------------------------------------------

    x = _flatten_feature(X_feat)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError(
            "La feature no contiene valores válidos."
        )

    # -----------------------------------------------------------------
    # BASIC STATS
    # -----------------------------------------------------------------

    mean_val = np.mean(x)

    std_val = np.std(x)

    median_val = np.median(x)

    min_val = np.min(x)

    max_val = np.max(x)

    p01 = np.percentile(x, 1)

    p99 = np.percentile(x, 99)

    dynamic_range = max_val - min_val

    # -----------------------------------------------------------------
    # SHAPE STATS
    # -----------------------------------------------------------------

    skewness = skew(x, bias=False)

    excess_kurtosis = kurtosis(
        x,
        fisher=True,
        bias=False,
    )

    # -----------------------------------------------------------------
    # NORMALITY TEST
    # -----------------------------------------------------------------

    normality_stat = None
    normality_p = None

    # scipy normaltest falla con n<8
    if len(x) >= 8:

        try:

            stat, pval = normaltest(x)

            normality_stat = stat
            normality_p = pval

        except Exception:
            pass

    # -----------------------------------------------------------------
    # OUTLIERS
    # -----------------------------------------------------------------

    outlier_fraction = _compute_outlier_fraction(x)

    # -----------------------------------------------------------------
    # CONSTANT FEATURE DETECTION
    # -----------------------------------------------------------------

    near_constant = bool(std_val < 1e-8)

    # -----------------------------------------------------------------
    # GAUSSIANITY SCORE
    # -----------------------------------------------------------------

    g_score = _gaussianity_score(
        skewness_abs=abs(skewness),
        kurtosis_abs=abs(excess_kurtosis),
        outlier_fraction=outlier_fraction,
        normality_p=normality_p,
    )

    g_label = _distribution_label(g_score)

    # -----------------------------------------------------------------
    # METADATA CONSISTENCY
    # -----------------------------------------------------------------

    declared_gaussian = None

    metadata_consistency = None

    if metadata is not None:

        declared_gaussian = bool(
            metadata.gaussian_friendly
        )

        empirical_gaussian = g_score >= 0.65

        metadata_consistency = (
            declared_gaussian == empirical_gaussian
        )

    # -----------------------------------------------------------------
    # WARNINGS
    # -----------------------------------------------------------------

    warnings = []

    if near_constant:
        warnings.append(
            "Feature casi constante."
        )

    if abs(skewness) > 2:
        warnings.append(
            "Alta asimetría (skewness)."
        )

    if abs(excess_kurtosis) > 5:
        warnings.append(
            "Kurtosis extrema."
        )

    if outlier_fraction > 0.05:
        warnings.append(
            "Muchos outliers."
        )

    if g_score < 0.40:
        warnings.append(
            "Distribución poco compatible con Gaussian HMM."
        )

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------

    report = {

        # -------------------------------------------------------------
        # SHAPE
        # -------------------------------------------------------------

        "n_samples": int(len(x)),

        # -------------------------------------------------------------
        # CENTRAL TENDENCY
        # -------------------------------------------------------------

        "mean": _safe_float(mean_val),
        "std": _safe_float(std_val),
        "median": _safe_float(median_val),

        # -------------------------------------------------------------
        # RANGE
        # -------------------------------------------------------------

        "min": _safe_float(min_val),
        "max": _safe_float(max_val),
        "p01": _safe_float(p01),
        "p99": _safe_float(p99),
        "dynamic_range": _safe_float(dynamic_range),

        # -------------------------------------------------------------
        # DISTRIBUTION SHAPE
        # -------------------------------------------------------------

        "skewness": _safe_float(skewness),
        "excess_kurtosis": _safe_float(excess_kurtosis),

        # -------------------------------------------------------------
        # NORMALITY
        # -------------------------------------------------------------

        "normality_test_stat": _safe_float(normality_stat),
        "normality_test_pvalue": _safe_float(normality_p),

        # -------------------------------------------------------------
        # OUTLIERS
        # -------------------------------------------------------------

        "outlier_fraction": _safe_float(outlier_fraction),

        # -------------------------------------------------------------
        # FLAGS
        # -------------------------------------------------------------

        "near_constant": near_constant,

        # -------------------------------------------------------------
        # GAUSSIANITY
        # -------------------------------------------------------------

        "gaussianity_score": _safe_float(g_score),
        "gaussianity_label": g_label,

        # -------------------------------------------------------------
        # METADATA CHECK
        # -------------------------------------------------------------

        "declared_gaussian_friendly": declared_gaussian,
        "metadata_consistency": metadata_consistency,

        # -------------------------------------------------------------
        # WARNINGS
        # -------------------------------------------------------------

        "warnings": warnings,
    }

    return report