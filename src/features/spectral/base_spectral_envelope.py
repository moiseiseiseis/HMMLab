# src/features/spectral/base_spectral_envelope.py

from __future__ import annotations

import numpy as np
import mne

from scipy import signal

from src.features.base import (
    BaseFeature,
    WindowArray,
    FeatureVector,
)


class BaseSpectralEnvelopeFeature(BaseFeature):
    """
    Clase base para envelopes espectrales vía:
        filtro banda + Hilbert envelope.

    Las subclases SOLO definen:
        - band_name
        - fmin
        - fmax
        - metadata
    """

    band_name: str
    fmin: float
    fmax: float

    def compute(
        self,
        window_data: WindowArray,
        fs: float,
        **kwargs
    ) -> FeatureVector:

        filtered = mne.filter.filter_data(
            window_data,
            sfreq=fs,
            l_freq=self.fmin,
            h_freq=self.fmax,
            method="iir",
            verbose=False,
        )

        envelope = np.abs(signal.hilbert(filtered))

        power = np.mean(envelope, axis=1)

        # Log estabiliza distribución
        power = np.log(power + 1e-12)

        return power.astype(np.float64)