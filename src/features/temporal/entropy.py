# src/features/temporal/entropy.py
"""
Entropía temporal por canal.

Implementación: Sample Entropy aproximada vía Shannon Entropy sobre
la distribución de amplitudes (histograma de 10 bins sobre la ventana).

¿Por qué esta aproximación y no Sample Entropy "verdadera"?
    Sample Entropy (SampEn) de Richman & Moorman es computacionalmente
    cara para el loop de ventanas (O(n²) por canal por ventana). Para un
    pipeline de HMM donde se procesan miles de ventanas por sujeto, la
    versión histograma es un balance razonable entre costo y información.

    Si en el futuro quieres Sample Entropy real, cambia el interior de
    compute() sin tocar ningún otro archivo. La interfaz es la misma.

Referencia Shannon Entropy para EEG:
    Inouye et al. (1991). Quantification of EEG irregularity by use of
    the entropy of the power spectrum. Electroencephalography and
    Clinical Neurophysiology, 79(3), 204-210.

Notas para HMM:
    La entropía temporal NO es gaussiana por defecto — tiene un ceiling
    natural (log(n_bins)) y suele ser left-skewed. Recomendamos robust_zscore
    antes del PCA. El diagnóstico D1 verificará esto empíricamente.
"""

from __future__ import annotations

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY


# ---------------------------------------------------------------------------
# Función de cómputo interna
# ---------------------------------------------------------------------------

def _shannon_entropy_histogram(signal_1d: np.ndarray, n_bins: int = 10) -> float:
    """
    Shannon Entropy de la distribución de amplitudes de una señal 1D.

    H = -Σ p_i * log2(p_i)   (solo sobre bins con p_i > 0)

    Rango: [0, log2(n_bins)]
        0          → señal completamente determinista (todos los samples en un bin)
        log2(n_bins) → distribución completamente uniforme sobre todos los bins

    Args:
        signal_1d: Array 1D de shape (n_samples,).
        n_bins:    Número de bins del histograma. Default 10.
                   Más bins → más sensible a detalles de la distribución.
                   Menos bins → más robusto a outliers.

    Returns:
        Valor de entropía en bits. 0.0 si la señal es plana.
    """
    if np.std(signal_1d) < 1e-12:
        # Señal plana: entropía = 0 (máxima certeza, mínima información)
        return 0.0

    counts, _ = np.histogram(signal_1d, bins=n_bins)
    probs = counts / counts.sum()

    # Solo considerar bins con probabilidad > 0 (log(0) no definido)
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    return float(entropy)


# ---------------------------------------------------------------------------
# TemporalEntropyFeature
# ---------------------------------------------------------------------------

@REGISTRY.register
class TemporalEntropyFeature(BaseFeature):
    """
    Entropía temporal por canal (Shannon Entropy sobre histograma de amplitudes).

    Captura la irregularidad/complejidad de la distribución de amplitudes
    dentro de una ventana. Alta entropía → distribución más uniforme (más
    irregular). Baja entropía → distribución concentrada (señal más regular,
    más predecible).

    Aplicaciones en EEG:
        - Diferencia entre estados de baja complejidad (ritmos sinusoidales
          dominantes, ej: sueño, reposo con ojos cerrados) y alta complejidad
          (actividad cognitiva, procesamiento activo).
        - Complementa Hjorth Complexity (que mide irregularidad frecuencial;
          la entropía mide irregularidad en amplitud).

    Output: (n_channels,) — un valor por canal.
    """

    # Número de bins del histograma. Atributo de clase para facilitar
    # futuras subclases con diferente n_bins si se necesita.
    N_BINS: int = 10

    metadata = FeatureMetadata(
        name="temporal_entropy",
        display_name="Temporal Entropy (Shannon, histogram)",
        category="temporal",
        gaussian_friendly=False,
        recommended_scaling="robust_zscore",
        expected_temporal_smoothness="low",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["hjorth_complexity"],
        window_sensitivity="high",
        subject_variability="medium",
        hmm_suitability="medium",
        hmm_risk_notes=(
            "La entropía tiene un ceiling natural en log2(N_BINS). "
            "Su distribución suele ser left-skewed (acumulación hacia valores altos). "
            "NO es gaussiana sin transformación. Usar robust_zscore. "
            "Alta sensibilidad al window_size: ventanas muy pequeñas producen "
            "estimaciones ruidosas del histograma. Recomendar window_size >= 200ms. "
            "Correlación esperada con hjorth_complexity — si es > 0.8, evaluar "
            "si vale la pena incluir ambas o solo una."
        ),
        n_channels_dependent=True,
    )

    def compute(self, window_data: WindowArray, fs: float, **kwargs) -> FeatureVector:
        """
        Args:
            window_data: (n_channels, n_samples)
            fs:          No se usa (la entropía es puramente temporal).

        Returns:
            Array de shape (n_channels,) con entropía por canal.
        """
        n_channels = window_data.shape[0]
        entropies = np.zeros(n_channels, dtype=np.float64)

        for ch in range(n_channels):
            entropies[ch] = _shannon_entropy_histogram(window_data[ch], self.N_BINS)

        return entropies
