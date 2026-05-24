# src/features/__init__.py
"""
Inicialización del sistema modular de features EEG-HMM.

IMPORTANTE:
    Este archivo DEBE importar todos los submódulos de features para que
    los decoradores @REGISTRY.register se ejecuten automáticamente.

    Si un módulo no se importa aquí:
        → su feature NO existe para el pipeline
        → REGISTRY aparecerá vacío o incompleto

El usuario solo necesita:
    import src.features

y todas las features quedan registradas.
"""

# ---------------------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------------------

from src.features.temporal.hjorth import (
    HjorthMobilityFeature,
    HjorthComplexityFeature,
)

from src.features.temporal.entropy import (
    TemporalEntropyFeature,
)

# ---------------------------------------------------------------------
# SPECTRAL FEATURES
# ---------------------------------------------------------------------

from src.features.spectral.alpha_envelope import (
    AlphaEnvelopeFeature,
)

from src.features.spectral.theta_envelope import (
    ThetaEnvelopeFeature,
)

from src.features.spectral.beta_envelope import (
    BetaEnvelopeFeature,
)

from src.features.spectral.relative_power import (
    RelativePowerFeature,
)

from src.features.spectral.band_ratios import (
    BandRatiosFeature,
)

# ---------------------------------------------------------------------
# SPATIAL FEATURES
# ---------------------------------------------------------------------

from src.features.spatial.hemispheric_asymmetry import (
    HemisphericAsymmetryFeature,
)

# ---------------------------------------------------------------------
# REGISTRY EXPORT
# ---------------------------------------------------------------------

from src.features.registry import REGISTRY

__all__ = [
    "REGISTRY",

    "HjorthMobilityFeature",
    "HjorthComplexityFeature",

    "TemporalEntropyFeature",

    "AlphaEnvelopeFeature",
    "ThetaEnvelopeFeature",
    "BetaEnvelopeFeature",

    "RelativePowerFeature",
    "BandRatiosFeature",

    "HemisphericAsymmetryFeature",
]