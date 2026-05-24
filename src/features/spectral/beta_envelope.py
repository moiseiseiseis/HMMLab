"""
Beta envelope feature usando Hilbert transform.
"""

from __future__ import annotations

from src.features.base import (
    BaseFeature,
    FeatureMetadata,
    WindowArray,
    FeatureVector,
)

from src.features.registry import REGISTRY

from src.features.spectral._hilbert_envelope import (
    compute_band_envelope,
)

# Banda beta estándar
BETA_LOW_HZ = 13.0
BETA_HIGH_HZ = 30.0


@REGISTRY.register
class BetaEnvelopeFeature(BaseFeature):

    metadata = FeatureMetadata(
        name="beta_envelope",
        display_name="Beta Envelope",
        category="spectral",

        gaussian_friendly=True,
        recommended_scaling="log_zscore",

        expected_temporal_smoothness="medium",
        expected_stationarity="quasi_stationary",

        window_sensitivity="high",
        subject_variability="high",

        hmm_suitability="medium",

        expected_redundancy_with=[],

        hmm_risk_notes=(
            "Beta envelope puede contaminarse fácilmente por EMG. "
            "Interpretar incrementos frontales con cuidado."
        ),

        n_channels_dependent=True,
    )

    def compute(
        self,
        window_data: WindowArray,
        fs: float,
        **kwargs,
    ) -> FeatureVector:

        return compute_band_envelope(
            window_data=window_data,
            fs=fs,
            low_hz=BETA_LOW_HZ,
            high_hz=BETA_HIGH_HZ,
            log_transform=True,
        )