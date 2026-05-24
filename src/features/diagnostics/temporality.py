# src/features/diagnostics/temporality.py
"""
Diagnósticos temporales para features EEG-HMM.

Objetivo:
Evaluar si las features tienen propiedades temporales compatibles
con modelos dinámicos tipo HMM.

Incluye:

1. Temporal smoothness
   - Autocorrelación lag-1
   - Variación entre ventanas consecutivas
   - Transition noise ratio

2. Stationarity
   - Drift temporal
   - Cambio de media inicio vs final
   - KPSS opcional (si statsmodels disponible)

3. Window sensitivity
   - Comparación entre distintos tamaños de ventana
   - Robustez estructural de la feature

IMPORTANTE:
Estos diagnósticos NO evalúan "verdad neurobiológica".
Evalúan compatibilidad geométrica/estadística con HMMs gaussianos.
"""

from __future__ import annotations

import warnings
import numpy as np

from dataclasses import dataclass
from typing import Optional

# KPSS opcional
try:
    from statsmodels.tsa.stattools import kpss
    HAS_KPSS = True
except Exception:
    HAS_KPSS = False


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class TemporalSmoothnessResult:
    feature_name: str

    lag1_autocorr_mean: float
    lag1_autocorr_std: float

    delta_mean: float
    delta_std: float

    transition_noise_ratio: float

    interpretation: str


@dataclass
class StationarityResult:
    feature_name: str

    drift_slope_mean: float
    drift_slope_std: float

    start_end_shift_mean: float

    kpss_pvalue_mean: Optional[float]

    interpretation: str


@dataclass
class WindowSensitivityResult:
    feature_name: str

    correlation_between_windows: float
    mean_absolute_difference: float

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


def _safe_autocorr(x: np.ndarray, lag: int = 1) -> float:
    """
    Autocorrelación segura.
    """
    if len(x) <= lag:
        return 0.0

    return _safe_corr(x[:-lag], x[lag:])


# ---------------------------------------------------------------------
# Temporal Smoothness
# ---------------------------------------------------------------------

def evaluate_temporal_smoothness(
    X: np.ndarray,
    feature_name: str,
) -> TemporalSmoothnessResult:
    """
    Evalúa suavidad temporal de una feature.

    Args:
        X:
            shape = (n_windows, n_dims)

    Returns:
        TemporalSmoothnessResult
    """

    if X.ndim != 2:
        raise ValueError("X debe tener shape (n_windows, n_dims)")

    autocorrs = []
    deltas = []

    for d in range(X.shape[1]):

        x = X[:, d]

        ac = _safe_autocorr(x, lag=1)
        autocorrs.append(ac)

        dx = np.diff(x)

        deltas.append(np.mean(np.abs(dx)))

    autocorrs = np.asarray(autocorrs)
    deltas = np.asarray(deltas)

    lag1_mean = float(np.mean(autocorrs))
    lag1_std = float(np.std(autocorrs))

    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas))

    # -------------------------------------------------------------
    # Noise ratio
    # -------------------------------------------------------------
    global_std = np.mean(np.std(X, axis=0)) + 1e-12
    transition_noise_ratio = delta_mean / global_std

    # -------------------------------------------------------------
    # Interpretación heurística
    # -------------------------------------------------------------
    if lag1_mean > 0.85:
        interpretation = "very_smooth"
    elif lag1_mean > 0.65:
        interpretation = "smooth"
    elif lag1_mean > 0.40:
        interpretation = "moderate"
    else:
        interpretation = "noisy"

    return TemporalSmoothnessResult(
        feature_name=feature_name,

        lag1_autocorr_mean=lag1_mean,
        lag1_autocorr_std=lag1_std,

        delta_mean=delta_mean,
        delta_std=delta_std,

        transition_noise_ratio=float(transition_noise_ratio),

        interpretation=interpretation,
    )


# ---------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------

def evaluate_stationarity(
    X: np.ndarray,
    feature_name: str,
) -> StationarityResult:
    """
    Evalúa estacionaridad temporal.

    Estrategia:
    - drift lineal
    - diferencia inicio vs final
    - KPSS opcional
    """

    if X.ndim != 2:
        raise ValueError("X debe tener shape (n_windows, n_dims)")

    n_windows = X.shape[0]

    t = np.arange(n_windows)

    slopes = []
    shifts = []
    kpss_pvalues = []

    for d in range(X.shape[1]):

        x = X[:, d]

        # ---------------------------------------------------------
        # Drift lineal
        # ---------------------------------------------------------
        try:
            slope = np.polyfit(t, x, deg=1)[0]
        except Exception:
            slope = 0.0

        slopes.append(slope)

        # ---------------------------------------------------------
        # Shift inicio-final
        # ---------------------------------------------------------
        first = np.mean(x[: max(5, n_windows // 10)])
        last = np.mean(x[-max(5, n_windows // 10):])

        shifts.append(abs(last - first))

        # ---------------------------------------------------------
        # KPSS
        # ---------------------------------------------------------
        if HAS_KPSS:

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    _, pvalue, _, _ = kpss(
                        x,
                        regression="c",
                        nlags="auto",
                    )

                kpss_pvalues.append(float(pvalue))

            except Exception:
                pass

    slope_mean = float(np.mean(slopes))
    slope_std = float(np.std(slopes))

    shift_mean = float(np.mean(shifts))

    if len(kpss_pvalues) > 0:
        kpss_mean = float(np.mean(kpss_pvalues))
    else:
        kpss_mean = None

    # -------------------------------------------------------------
    # Interpretación
    # -------------------------------------------------------------
    abs_slope = abs(slope_mean)

    if abs_slope < 1e-4 and shift_mean < 0.1:
        interpretation = "stationary"

    elif abs_slope < 1e-3:
        interpretation = "quasi_stationary"

    else:
        interpretation = "non_stationary"

    return StationarityResult(
        feature_name=feature_name,

        drift_slope_mean=slope_mean,
        drift_slope_std=slope_std,

        start_end_shift_mean=shift_mean,

        kpss_pvalue_mean=kpss_mean,

        interpretation=interpretation,
    )


# ---------------------------------------------------------------------
# Window Sensitivity
# ---------------------------------------------------------------------

def evaluate_window_sensitivity(
    X_small: np.ndarray,
    X_large: np.ndarray,
    feature_name: str,
) -> WindowSensitivityResult:
    """
    Evalúa sensibilidad a tamaño de ventana.

    Args:
        X_small:
            Feature extraída con ventana pequeña

        X_large:
            Feature extraída con ventana grande

    Importante:
    Ambos arrays deben representar la MISMA sesión EEG.
    """

    if X_small.ndim != 2 or X_large.ndim != 2:
        raise ValueError("Inputs deben ser 2D")

    # -------------------------------------------------------------
    # Igualar longitud
    # -------------------------------------------------------------
    n = min(len(X_small), len(X_large))

    Xs = X_small[:n]
    Xl = X_large[:n]

    # -------------------------------------------------------------
    # Flatten
    # -------------------------------------------------------------
    xs = Xs.ravel()
    xl = Xl.ravel()

    corr = _safe_corr(xs, xl)

    mad = float(np.mean(np.abs(xs - xl)))

    # -------------------------------------------------------------
    # Interpretación
    # -------------------------------------------------------------
    if corr > 0.90:
        interpretation = "robust"

    elif corr > 0.70:
        interpretation = "moderate"

    else:
        interpretation = "sensitive"

    return WindowSensitivityResult(
        feature_name=feature_name,

        correlation_between_windows=float(corr),

        mean_absolute_difference=mad,

        interpretation=interpretation,
    )

# ---------------------------------------------------------------------
# MASTER TEMPORAL DIAGNOSTIC
# ---------------------------------------------------------------------

def compute_temporal_diagnostics(
    X: np.ndarray,
    subject_slices: dict | None = None,
) -> dict:
    """
    Ejecuta todos los diagnósticos temporales.

    Parameters
    ----------
    X : np.ndarray
        shape = (n_windows, n_dims)

    subject_slices : dict
        Actualmente no usado explícitamente,
        pero se deja para futuras extensiones
        intra-sujeto.

    Returns
    -------
    dict
    """

    smoothness = evaluate_temporal_smoothness(
        X,
        feature_name="feature",
    )

    stationarity = evaluate_stationarity(
        X,
        feature_name="feature",
    )

    return {
        "smoothness": {
            "lag1_autocorr_mean": smoothness.lag1_autocorr_mean,
            "lag1_autocorr_std": smoothness.lag1_autocorr_std,
            "delta_mean": smoothness.delta_mean,
            "delta_std": smoothness.delta_std,
            "transition_noise_ratio": smoothness.transition_noise_ratio,
            "interpretation": smoothness.interpretation,
        },

        "stationarity": {
            "drift_slope_mean": stationarity.drift_slope_mean,
            "drift_slope_std": stationarity.drift_slope_std,
            "start_end_shift_mean": stationarity.start_end_shift_mean,
            "kpss_pvalue_mean": stationarity.kpss_pvalue_mean,
            "interpretation": stationarity.interpretation,
        },
    }