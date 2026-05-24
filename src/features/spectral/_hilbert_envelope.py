"""
Utilities para cálculo de envelopes espectrales vía Hilbert transform.
"""

from __future__ import annotations

import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import hilbert


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Bandpass Butterworth zero-phase.

    Soporta:
        - 1D: (n_samples,)
        - 2D: (n_channels, n_samples)
    """

    nyquist = fs / 2.0

    low = low_hz / nyquist
    high = high_hz / nyquist

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal, axis=-1)


def compute_band_envelope(
    window_data: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Calcula envelope espectral promedio vía Hilbert transform.

    INPUT:
        1D:
            (n_samples,)

        2D:
            (n_channels, n_samples)

    OUTPUT:
        1D:
            (1,) si input 1D
            (n_channels,) si input 2D
    """

    # ---------------------------------------------------------
    # Bandpass
    # ---------------------------------------------------------

    filtered = bandpass_filter(
        signal=window_data,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )

    # ---------------------------------------------------------
    # Hilbert transform
    # ---------------------------------------------------------

    analytic = hilbert(filtered, axis=-1)

    envelope = np.abs(analytic)

    # ---------------------------------------------------------
    # Mean amplitude
    # ---------------------------------------------------------

    if envelope.ndim == 1:
        values = np.array([np.mean(envelope)])

    else:
        values = np.mean(envelope, axis=1)

    # ---------------------------------------------------------
    # Log-transform
    # ---------------------------------------------------------

    if log_transform:
        values = np.log1p(values)

    return values.astype(np.float64)