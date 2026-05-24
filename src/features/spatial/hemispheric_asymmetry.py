# src/features/spatial/hemispheric_asymmetry.py
"""
Asimetría hemisférica por par de canales y banda espectral.

Cuantifica el desequilibrio izquierda-derecha en potencia espectral:

    asym(par, banda) = (R - L) / (R + L + 1e-10)

Resultado en (-1, 1):
    -1 → toda la potencia en hemisferio izquierdo
     0 → simetría perfecta
    +1 → toda la potencia en hemisferio derecho

Pares evaluados (7):
    F3-F4, C3-C4, P3-P4, O1-O2, T3-T4, T5-T6, F7-F8

Dependencias:
    Requiere al menos una banda activa en `envs` (alpha, theta y/o beta).
    Si ninguna está activa → warning + array vacío shape (0,).

Notas para HMM:
    La asimetría alpha frontal (F3-F4 alpha) es el correlato de motivación
    y procesamiento emocional más replicado en EEG (Davidson, 1988).
    Valores fuertemente asimétricos pueden indicar artefactos asimétricos
    (EOG en frontales) — verificar con QC antes de incluir en modelo.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.features.base import BaseFeature, FeatureMetadata, WindowArray, FeatureVector
from src.features.registry import REGISTRY

# -------------------------------------------------------------------------
# Montaje y pares
# -------------------------------------------------------------------------

_CHS: list[str] = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2',
]
_CH_IDX: dict[str, int] = {ch: i for i, ch in enumerate(_CHS)}

# (canal_izquierdo, canal_derecho) — orden canónico
_PAIRS: list[tuple[str, str]] = [
    ('F3', 'F4'),
    ('C3', 'C4'),
    ('P3', 'P4'),
    ('O1', 'O2'),
    ('T3', 'T4'),
    ('T5', 'T6'),
    ('F7', 'F8'),
]
_N_PAIRS = len(_PAIRS)

# Índices pre-calculados para acceso vectorizado
_L_IDX: np.ndarray = np.array([_CH_IDX[L] for L, _ in _PAIRS], dtype=np.intp)
_R_IDX: np.ndarray = np.array([_CH_IDX[R] for _, R in _PAIRS], dtype=np.intp)

# Orden en que se procesan las bandas (solo las presentes en envs)
_BAND_ORDER: tuple[str, ...] = ('alpha', 'theta', 'beta')


@REGISTRY.register
class HemisphericAsymmetryFeature(BaseFeature):
    """
    Asimetría hemisférica (R-L)/(R+L+eps) por par y banda.

    Output por ventana: (7 * n_bandas_activas,)
    Orden: [F3F4_b0, C3C4_b0, ..., F7F8_b0, F3F4_b1, ...]
    donde b0 < b1 < b2 sigue el orden alpha → theta → beta.

    Si ninguna banda está activa: array vacío shape (0,) + warning.
    """

    metadata = FeatureMetadata(
        name="hemispheric_asymmetry",
        display_name="Hemispheric Asymmetry (7 pairs × active bands)",
        category="connectivity",
        gaussian_friendly=True,
        recommended_scaling="zscore",
        expected_temporal_smoothness="medium",
        expected_stationarity="quasi_stationary",
        expected_redundancy_with=["alpha_envelope", "theta_envelope", "beta_envelope"],
        window_sensitivity="medium",
        subject_variability="high",
        hmm_suitability="medium",
        hmm_risk_notes=(
            "La asimetría frontal alpha (F3-F4) es sensible a EOG y artefactos "
            "musculares asimétricos. Alta variabilidad inter-sujeto en reposo. "
            "Verificar QC de absorbentes antes de incluir en modelo. "
            "Output de dimensión dinámica: depende de cuántas bandas estén activas. "
            "Requiere al menos una de alpha/theta/beta en el YAML."
        ),
        n_channels_dependent=False,
    )

    def output_dim(self, n_channels: int) -> int:
        # Dimensión dinámica — no conocida sin ver envs.
        # validate_output es sobreescrito para evitar chequeo de forma.
        return _N_PAIRS  # valor de fallback; no se usa en la validación real

    def validate_output(self, result: FeatureVector, n_channels: int) -> FeatureVector:
        """
        Sobreescribe validate_output para omitir el chequeo de forma.
        La dimensión es dinámica: 7 × n_bandas_activas.
        Solo sanitiza NaN/Inf.
        """
        result = np.atleast_1d(result).ravel().astype(np.float64)
        return np.where(np.isfinite(result), result, 0.0)

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
            envs:        Dict con alguna combinación de 'alpha', 'theta', 'beta',
                         cada array de shape (n_channels,) con n_channels >= 19.

        Returns:
            Array (7 * n_bandas_activas,).
            Si ninguna banda activa → array vacío shape (0,) + warning.
        """
        envs = envs or {}

        active_bands = [b for b in _BAND_ORDER if b in envs]
        if not active_bands:
            warnings.warn(
                "hemispheric_asymmetry: ninguna banda activa en envs "
                "(se necesita alpha, theta y/o beta). "
                "Devolviendo array vacío. "
                "Activa al menos una banda espectral en el YAML.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.array([], dtype=np.float64)

        parts: list[np.ndarray] = []
        for band in active_bands:
            env = envs[band]                    # (n_channels,)
            L = env[_L_IDX]                     # (7,)
            R = env[_R_IDX]                     # (7,)
            asym = (R - L) / (R + L + 1e-10)   # (7,) — garantizado en (-1, 1)
            parts.append(asym)

        return np.concatenate(parts)            # (7 * n_active_bands,)
