# src/features/spectral/theta_envelope.py

from __future__ import annotations

from src.features.base import FeatureMetadata
from src.features.registry import REGISTRY

from src.features.spectral.base_spectral_envelope import (
    BaseSpectralEnvelopeFeature,
)


@REGISTRY.register
class ThetaEnvelopeFeature(BaseSpectralEnvelopeFeature):

    band_name = "theta"

    fmin = 4.0
    fmax = 8.0

    metadata = FeatureMetadata(
        name="theta_envelope",
        display_name="Theta Envelope",
        category="spectral",

        gaussian_friendly=True,
        recommended_scaling="zscore",

        expected_temporal_smoothness="high",
        expected_stationarity="quasi_stationary",

        expected_redundancy_with=[],

        window_sensitivity="high",
        subject_variability="medium",

        hmm_suitability="high",

        hmm_risk_notes=(
            "Theta envelope puede contaminarse por artefactos oculares "
            "frontales y drift lento."
        ),

        n_channels_dependent=True,
    )