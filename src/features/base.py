# src/features/base.py
"""
Núcleo del sistema de features.

Define:
  - FeatureMetadata: metadata declarativa de una feature
  - BaseFeature: clase base que toda feature debe heredar

NADA en este archivo hace cómputo científico.
NADA en este archivo sabe de MNE/scipy.
"""

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------
# TYPE ALIASES
# ---------------------------------------------------------------------

WindowArray = np.ndarray
FeatureVector = np.ndarray


# ---------------------------------------------------------------------
# FEATURE METADATA
# ---------------------------------------------------------------------

@dataclass
class FeatureMetadata:
    """
    Metadata declarativa de una feature EEG.

    IMPORTANTE:
    En dataclasses de Python TODOS los campos sin default
    deben ir antes de cualquier campo con default.
    """

    # -----------------------------------------------------------------
    # IDENTIDAD
    # -----------------------------------------------------------------

    name: str
    display_name: str

    category: Literal[
        "spectral",
        "temporal",
        "global_field",
        "connectivity",
        "ratio",
    ]

    # -----------------------------------------------------------------
    # PROPIEDADES ESTADÍSTICAS
    # -----------------------------------------------------------------

    gaussian_friendly: bool

    recommended_scaling: Literal[
        "zscore",
        "log_zscore",
        "robust_zscore",
        "none",
    ]

    # -----------------------------------------------------------------
    # PROPIEDADES TEMPORALES
    # -----------------------------------------------------------------

    expected_temporal_smoothness: Literal[
        "low",
        "medium",
        "high",
    ]

    expected_stationarity: Literal[
        "stationary",
        "quasi_stationary",
        "non_stationary",
    ]

    # -----------------------------------------------------------------
    # SENSIBILIDAD / VARIABILIDAD
    # -----------------------------------------------------------------

    window_sensitivity: Literal[
        "low",
        "medium",
        "high",
    ]

    subject_variability: Literal[
        "low",
        "medium",
        "high",
    ]

    # -----------------------------------------------------------------
    # HMM SUITABILITY
    # -----------------------------------------------------------------

    hmm_suitability: Literal[
        "low",
        "medium",
        "high",
    ]

    # =================================================================
    # CAMPOS OPCIONALES (DEBEN IR AL FINAL)
    # =================================================================

    expected_redundancy_with: list[str] = field(default_factory=list)

    hmm_risk_notes: str = ""

    n_channels_dependent: bool = True

    # -----------------------------------------------------------------
    # VALIDACIÓN
    # -----------------------------------------------------------------

    def __post_init__(self):

        if not self.name:
            raise ValueError(
                "FeatureMetadata.name no puede estar vacío."
            )

        if " " in self.name:
            raise ValueError(
                f"FeatureMetadata.name debe ser snake_case. "
                f"Recibido: '{self.name}'"
            )

        if not self.display_name:
            raise ValueError(
                "FeatureMetadata.display_name no puede estar vacío."
            )

    # -----------------------------------------------------------------
    # SERIALIZACIÓN
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:

        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "gaussian_friendly": self.gaussian_friendly,
            "recommended_scaling": self.recommended_scaling,
            "expected_temporal_smoothness": self.expected_temporal_smoothness,
            "expected_stationarity": self.expected_stationarity,
            "window_sensitivity": self.window_sensitivity,
            "subject_variability": self.subject_variability,
            "hmm_suitability": self.hmm_suitability,
            "expected_redundancy_with": self.expected_redundancy_with,
            "hmm_risk_notes": self.hmm_risk_notes,
            "n_channels_dependent": self.n_channels_dependent,
        }


# ---------------------------------------------------------------------
# BASE FEATURE
# ---------------------------------------------------------------------

class BaseFeature(ABC):
    """
    Clase base para todas las features del pipeline EEG-HMM.
    """

    metadata: FeatureMetadata

    @abstractmethod
    def compute(
        self,
        window_data: WindowArray,
        fs: float,
        **kwargs,
    ) -> FeatureVector:
        """
        Computa el vector de features para una ventana EEG.

        Args:
            window_data:
                Shape (n_channels, n_samples)

            fs:
                Frecuencia de muestreo

        Returns:
            np.ndarray 1D
        """
        ...

    # -----------------------------------------------------------------
    # OUTPUT DIM
    # -----------------------------------------------------------------

    def output_dim(self, n_channels: int) -> int:

        if self.metadata.n_channels_dependent:
            return n_channels

        return 1

    # -----------------------------------------------------------------
    # VALIDACIÓN DE OUTPUT
    # -----------------------------------------------------------------

    def validate_output(
        self,
        result: FeatureVector,
        n_channels: int,
    ) -> FeatureVector:

        result = np.atleast_1d(result).ravel().astype(np.float64)

        expected_dim = self.output_dim(n_channels)

        if result.shape[0] != expected_dim:

            raise ValueError(
                f"Feature '{self.metadata.name}' devolvió "
                f"shape incorrecto.\n"
                f"Esperado: ({expected_dim},)\n"
                f"Recibido: {result.shape}"
            )

        # Sanitizar NaN/Inf
        result = np.where(
            np.isfinite(result),
            result,
            0.0,
        )

        return result

    # -----------------------------------------------------------------
    # REPRESENTACIÓN
    # -----------------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"name='{self.metadata.name}', "
            f"category='{self.metadata.category}', "
            f"hmm_suitability='{self.metadata.hmm_suitability}')"
        )

    def __str__(self) -> str:
        return self.metadata.display_name