# src/features/spectral/alpha_envelope.py
"""
Alpha Envelope (8–13 Hz) por canal, vía filtro Butterworth + Hilbert.

Banda alpha: 8–13 Hz
    El ritmo alfa es el oscilador cortical dominante en EEG humano.
    Emerge principalmente en regiones occipitales durante reposo con
    ojos cerrados, y se suprime (desincronización alfa / ERD) durante
    procesamiento visual y cognitivo activo.

    Su envelope captura la dinámica de potencia alfa, que es uno de
    los correlatos neuronales más robustos y reproducibles en EEG.

Notas para HMM:
    Los envelopes espectrales son log-normales por naturaleza (la potencia
    EEG sigue distribución 1/f con fluctuaciones multiplicativas). Usar
    log_zscore antes del PCA es crítico para que el PCA no sea dominado
    por sujetos con alta potencia alfa global.
"""

from __future__ import annotations

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY
from src.features.spectral._hilbert_envelope import compute_band_envelope

# Banda alpha estándar (Hz)
ALPHA_LOW_HZ  = 8.0
ALPHA_HIGH_HZ = 13.0


@REGISTRY.register
class AlphaEnvelopeFeature(BaseFeature):
    """
    Envelope de amplitud media en banda alpha (8–13 Hz) por canal.

    Output: (n_channels,) — un valor por canal en µV (si la entrada está en µV).
    """

    metadata = FeatureMetadata(
        name="alpha_envelope",
        display_name="Alpha Envelope (8–13 Hz, Hilbert)",
        category="spectral",
        gaussian_friendly=False,
        recommended_scaling="log_zscore",
        expected_temporal_smoothness="high",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["rms", "gfp"],
        window_sensitivity="medium",
        subject_variability="high",
        hmm_suitability="medium",
        hmm_risk_notes=(
            "Alta variabilidad inter-sujeto: la potencia alfa varía hasta 10x "
            "entre individuos. SIN log_zscore, el PCA será dominado por "
            "sujetos con alta potencia alfa. "
            "Riesgo de que el HMM aprenda 'sujeto A vs sujeto B' en lugar de "
            "estados cognitivos. Considerar demeaning por sujeto antes del PCA. "
            "Correlación esperada con rms y gfp (todas capturan amplitud global)."
        ),
        n_channels_dependent=True,
    )

    def compute(self, window_data: WindowArray, fs: float, **kwargs) -> FeatureVector:
        """
        Args:
            window_data: (n_channels, n_samples)
            fs:          Frecuencia de muestreo (Hz). Requerida para el filtro.

        Returns:
            Array de shape (n_channels,) con envelope alpha medio por canal.
        """
        n_channels = window_data.shape[0]
        envelopes = np.zeros(n_channels, dtype=np.float64)

        for ch in range(n_channels):
            envelopes[ch] = compute_band_envelope(
                window_data[ch], fs,
                low_hz=ALPHA_LOW_HZ,
                high_hz=ALPHA_HIGH_HZ,
            )

        return envelopes
