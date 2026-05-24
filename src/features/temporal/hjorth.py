# src/features/temporal/hjorth.py
"""
Parámetros de Hjorth: Mobility y Complexity.

Referencia:
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

Definiciones:
    Dado una señal x(t) con varianza Var(x), primera diferencia x'(t) con
    varianza Var(x'), y segunda diferencia x''(t) con varianza Var(x''):

    Activity   = Var(x)                          ← no se usa aquí, pero es la base
    Mobility   = sqrt(Var(x') / Var(x))          ← proxy de frecuencia media
    Complexity = sqrt(Var(x'') / Var(x')) / Mobility  ← proxy de irregularidad

Notas para HMM:
    Mobility y Complexity son features temporales que no requieren transformada
    de Fourier. Son relativamente robustas a la elección del tamaño de ventana
    (comparado con envelopes espectrales). Su distribución es aproximadamente
    gaussiana después de z-score, lo que los hace buenos candidatos para
    Gaussian HMM emission models.

    Riesgo principal: contaminación por EMG. El músculo elevará Mobility
    significativamente (el EMG tiene frecuencia media ~80-200 Hz).
    Asegurarse de que el preprocessing incluya filtro pasa-bajos antes de
    usar estas features.
"""

from __future__ import annotations

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY


# ---------------------------------------------------------------------------
# Función de cómputo compartida (privada al módulo)
# ---------------------------------------------------------------------------

def _hjorth_params(signal_1d: np.ndarray) -> tuple[float, float]:
    """
    Computa Mobility y Complexity de Hjorth para una señal 1D.

    Esta función es privada al módulo — no forma parte de la API pública.
    Está aquí para evitar duplicar la lógica entre Mobility y Complexity,
    que comparten los mismos cómputos intermedios.

    Args:
        signal_1d: Array 1D de shape (n_samples,).

    Returns:
        Tupla (mobility, complexity).
        Si la señal es plana (varianza = 0), devuelve (0.0, 0.0).
        Si la primera diferencia es plana, complexity = 0.0.
    """
    var_x = np.var(signal_1d)

    # Señal completamente plana — no se puede computar nada significativo.
    # Devolver 0.0 en lugar de NaN para no contaminar el PCA.
    if var_x < 1e-12:
        return 0.0, 0.0

    diff1 = np.diff(signal_1d)
    var_d1 = np.var(diff1)

    mobility = np.sqrt(var_d1 / var_x)

    # Primera diferencia plana — complexity no está definida.
    if var_d1 < 1e-12 or mobility < 1e-12:
        return mobility, 0.0

    diff2 = np.diff(diff1)
    var_d2 = np.var(diff2)

    complexity = (np.sqrt(var_d2 / var_d1)) / mobility

    return float(mobility), float(complexity)


# ---------------------------------------------------------------------------
# HjorthMobilityFeature
# ---------------------------------------------------------------------------

@REGISTRY.register
class HjorthMobilityFeature(BaseFeature):
    """
    Hjorth Mobility por canal.

    Mobility ≈ frecuencia media dominante de la señal (en unidades de
    frecuencia normalizada por la frecuencia de muestreo).

    Aplicaciones en EEG:
        - Diferencia entre estados de reposo (baja mobility, dominancia alfa)
          y estados activos (alta mobility, más beta/gamma).
        - Sensible a cambios en la frecuencia dominante durante tareas cognitivas.

    Output: (n_channels,) — un valor por canal.
    """

    metadata = FeatureMetadata(
        name="hjorth_mobility",
        display_name="Hjorth Mobility",
        category="temporal",
        gaussian_friendly=True,
        recommended_scaling="zscore",
        expected_temporal_smoothness="medium",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["hjorth_complexity"],
        window_sensitivity="medium",
        subject_variability="medium",
        hmm_suitability="high",
        hmm_risk_notes=(
            "Mobility captura proxy de frecuencia media. Sensible a EMG: "
            "la contaminación muscular eleva el valor significativamente. "
            "Requiere filtro pasa-bajos (típicamente 40-80 Hz) antes del cómputo. "
            "Correlación esperada con hjorth_complexity (ambas miden complejidad "
            "espectral) — verificar en diagnóstico D3 que no sean redundantes."
        ),
        n_channels_dependent=True,
    )

    def compute(self, window_data: WindowArray, fs: float, **kwargs) -> FeatureVector:
        """
        Args:
            window_data: (n_channels, n_samples)
            fs:          No se usa directamente, pero se pasa por consistencia.

        Returns:
            Array de shape (n_channels,) con Mobility por canal.
        """
        n_channels = window_data.shape[0]
        mobilities = np.zeros(n_channels, dtype=np.float64)

        for ch in range(n_channels):
            mobility, _ = _hjorth_params(window_data[ch])
            mobilities[ch] = mobility

        return mobilities


# ---------------------------------------------------------------------------
# HjorthComplexityFeature
# ---------------------------------------------------------------------------

@REGISTRY.register
class HjorthComplexityFeature(BaseFeature):
    """
    Hjorth Complexity por canal.

    Complexity mide cuánto se desvía la forma de onda de una oscilación
    sinusoidal pura. Una sinusoide perfecta tiene complexity = 1.
    Señales más irregulares (más ruido, más componentes frecuenciales) tienen
    complexity > 1.

    Aplicaciones en EEG:
        - Diferencia entre señales dominadas por ritmos regulares (alpha, mu)
          y señales más irregulares (broadband durante procesamiento cognitivo).
        - Complementa a Mobility: la combinación de ambos es más informativa
          que cada uno por separado.

    Output: (n_channels,) — un valor por canal.
    """

    metadata = FeatureMetadata(
        name="hjorth_complexity",
        display_name="Hjorth Complexity",
        category="temporal",
        gaussian_friendly=True,
        recommended_scaling="zscore",
        expected_temporal_smoothness="medium",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["hjorth_mobility"],
        window_sensitivity="medium",
        subject_variability="medium",
        hmm_suitability="medium",
        hmm_risk_notes=(
            "Complexity puede ser sensible a artefactos de movimiento y a "
            "discontinuidades entre épocas (si las ventanas cruzan bordes de época). "
            "Verificar que el step_size no genere ventanas que crucen eventos. "
            "Correlación esperada con hjorth_mobility — si la correlación empírica "
            "es > 0.85, considerar usar solo Mobility para evitar redundancia en PCA."
        ),
        n_channels_dependent=True,
    )

    def compute(self, window_data: WindowArray, fs: float, **kwargs) -> FeatureVector:
        """
        Args:
            window_data: (n_channels, n_samples)
            fs:          No se usa directamente.

        Returns:
            Array de shape (n_channels,) con Complexity por canal.
        """
        n_channels = window_data.shape[0]
        complexities = np.zeros(n_channels, dtype=np.float64)

        for ch in range(n_channels):
            _, complexity = _hjorth_params(window_data[ch])
            complexities[ch] = complexity

        return complexities
