# src/features/spectral/band_ratios.py
"""
Log-ratios entre bandas espectrales por canal.

Los log-ratios son preferibles a los ratios lineales porque:
  - Son simétricos: log(X/Y) = -log(Y/X)
  - Tienen distribución más cercana a gaussiana
  - Equivalen a diferencias en escala log, más estables numéricamente

Ratios calculados (por canal):
  theta/alpha : correlato clásico de carga cognitiva (memoria de trabajo,
                atención sostenida). Aumenta durante esfuerzo cognitivo.
  alpha/beta  : inversamente relacionado con activación cortical.
                Desciende durante procesamiento activo.
  theta/beta  : combinación de los dos anteriores.

Dependencias:
    Requiere alpha_envelope, theta_envelope, beta_envelope pre-calculados.
    No refiltrar — recibe arrays via `envs`.

Notas para HMM:
    log(theta/beta) = log(theta/alpha) + log(alpha/beta) — los tres ratios
    no son independientes. Incluir los tres en PCA con los envelopes base
    introduce redundancia sustancial. Verificar correlaciones en diagnóstico D3.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY

_N_RATIOS = 3   # log(θ/α), log(α/β), log(θ/β)
_BANDS    = ('alpha', 'theta', 'beta')


@REGISTRY.register
class BandRatiosFeature(BaseFeature):
    """
    Log-ratios theta/alpha, alpha/beta, theta/beta por canal.

    Output por ventana: (n_channels * 3,)
    Orden: [log_ta_ch0..n-1 | log_ab_ch0..n-1 | log_tb_ch0..n-1]
    """

    metadata = FeatureMetadata(
        name="band_ratios",
        display_name="Log Band Ratios (theta/alpha, alpha/beta, theta/beta)",
        category="ratio",
        gaussian_friendly=True,
        recommended_scaling="zscore",
        expected_temporal_smoothness="medium",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=[
            "relative_power", "alpha_envelope", "theta_envelope", "beta_envelope"
        ],
        window_sensitivity="medium",
        subject_variability="medium",
        hmm_suitability="high",
        hmm_risk_notes=(
            "theta/alpha es el biomarcador cognitivo más establecido en literatura. "
            "Los tres ratios no son linealmente independientes: "
            "log(θ/β) = log(θ/α) + log(α/β). "
            "Considerar usar solo log(θ/α) y log(α/β) para evitar redundancia en PCA. "
            "Requiere alpha, theta, beta envelopes activos."
        ),
        n_channels_dependent=True,
    )

    def output_dim(self, n_channels: int) -> int:
        return n_channels * _N_RATIOS

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
            Array (n_channels * 3,) con log-ratios.
            Garantizados finitos: el offset 1e-10 evita log(0).
        """
        n_channels = window_data.shape[0]
        envs = envs or {}

        missing = [b for b in _BANDS if b not in envs]
        if missing:
            warnings.warn(
                f"band_ratios: faltan bandas {missing} en envs. "
                f"Se usarán zeros para esas bandas. "
                f"Activa alpha/theta/beta envelopes en el YAML.",
                RuntimeWarning,
                stacklevel=2,
            )

        alpha = envs.get('alpha', np.zeros(n_channels, dtype=np.float64))
        theta = envs.get('theta', np.zeros(n_channels, dtype=np.float64))
        beta  = envs.get('beta',  np.zeros(n_channels, dtype=np.float64))

        log_alpha = np.log(np.abs(alpha) + 1e-10)
        log_theta = np.log(np.abs(theta) + 1e-10)
        log_beta  = np.log(np.abs(beta)  + 1e-10)

        log_theta_alpha = log_theta - log_alpha
        log_alpha_beta  = log_alpha - log_beta
        log_theta_beta  = log_theta - log_beta

        return np.concatenate([log_theta_alpha, log_alpha_beta, log_theta_beta])
