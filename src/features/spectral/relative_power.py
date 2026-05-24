# src/features/spectral/relative_power.py
"""
Potencia relativa de banda por canal.

Expresa la potencia de cada banda (alpha, theta, beta) como fracción de la
potencia total: x_rel = x / (alpha + theta + beta + 1e-10).

La suma de las tres fracciones por canal es siempre 1.0, lo que elimina
la variabilidad inter-sujeto de amplitud absoluta.

Dependencias:
    Requiere alpha_envelope, theta_envelope, beta_envelope activos en el
    pipeline. Recibe los arrays pre-calculados via `envs` — no refiltrar.

Notas para HMM:
    La normalización relativa hace que el modelo aprenda distribución espectral
    en lugar de escala de amplitud. Útil para comparar sujetos con distinta
    potencia EEG global. Por construcción, las tres fracciones suman 1 por
    canal — incluir las tres introduce colinealidad perfecta. Considerar usar
    solo alpha_rel y theta_rel en PCA (beta_rel queda determinada).
"""

from __future__ import annotations

import warnings

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY

_N_BANDS = 3   # alpha, theta, beta
_BANDS   = ('alpha', 'theta', 'beta')


@REGISTRY.register
class RelativePowerFeature(BaseFeature):
    """
    Potencia relativa de alpha, theta y beta por canal.

    Output por ventana: (n_channels * 3,)
    Orden: [alpha_rel_ch0..n-1 | theta_rel_ch0..n-1 | beta_rel_ch0..n-1]
    """

    metadata = FeatureMetadata(
        name="relative_power",
        display_name="Relative Band Power (alpha, theta, beta)",
        category="spectral",
        gaussian_friendly=False,
        recommended_scaling="zscore",
        expected_temporal_smoothness="high",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["alpha_envelope", "theta_envelope", "beta_envelope"],
        window_sensitivity="medium",
        subject_variability="low",
        hmm_suitability="high",
        hmm_risk_notes=(
            "Colinealidad perfecta: alpha_rel + theta_rel + beta_rel = 1 por canal. "
            "Incluir las tres en PCA añade una dimensión linealmente dependiente — "
            "considerar eliminar beta_rel. "
            "Requiere alpha, theta, beta envelopes activos. "
            "Sin ellos, devuelve zeros con warning."
        ),
        n_channels_dependent=True,
    )

    def output_dim(self, n_channels: int) -> int:
        return n_channels * _N_BANDS

    def compute(
        self,
        window_data: WindowArray,
        fs: float,
        envs: dict | None = None,
        **kwargs,
    ) -> FeatureVector:
        """
        Args:
            window_data: (n_channels, n_samples) — no se usa directamente.
            fs:          Frecuencia de muestreo (no se usa).
            envs:        Dict {'alpha': arr, 'theta': arr, 'beta': arr},
                         cada arr de shape (n_channels,).
                         Bandas faltantes → warning + zeros.

        Returns:
            Array (n_channels * 3,) con potencias relativas.
        """
        n_channels = window_data.shape[0]
        envs = envs or {}

        missing = [b for b in _BANDS if b not in envs]
        if missing:
            warnings.warn(
                f"relative_power: faltan bandas {missing} en envs. "
                f"Se usarán zeros para esas bandas. "
                f"Activa alpha/theta/beta envelopes en el YAML.",
                RuntimeWarning,
                stacklevel=2,
            )

        alpha = envs.get('alpha', np.zeros(n_channels, dtype=np.float64))
        theta = envs.get('theta', np.zeros(n_channels, dtype=np.float64))
        beta  = envs.get('beta',  np.zeros(n_channels, dtype=np.float64))

        total = alpha + theta + beta + 1e-10

        return np.concatenate([
            alpha / total,
            theta / total,
            beta  / total,
        ])
