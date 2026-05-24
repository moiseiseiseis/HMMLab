# src/features/diagnostics/hmm_score.py

"""
Heuristic HMM suitability scoring para features EEG.

Objetivo:
Combinar múltiples diagnósticos en un score interpretable
de compatibilidad con Gaussian HMMs.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def compute_hmm_suitability_score(
    distribution_report: dict,
    temporal_report: dict,
    inter_subject_report: dict,
    metadata=None,
) -> dict:
    """
    Combina diagnósticos en score heurístico global.
    """

    # =========================================================
    # DISTRIBUTION
    # =========================================================

    gaussianity = distribution_report.get(
        "gaussianity_score",
        0.5,
    )

    gaussianity = _clip01(gaussianity)

    # =========================================================
    # TEMPORAL
    # =========================================================

    smoothness = temporal_report.get(
        "smoothness",
        {},
    )

    lag1 = smoothness.get(
        "lag1_autocorr_mean",
        0.0,
    )

    lag1 = _clip01((lag1 + 1.0) / 2.0)

    noise_ratio = smoothness.get(
        "transition_noise_ratio",
        1.0,
    )

    temporal_score = lag1 * np.exp(-noise_ratio)

    temporal_score = _clip01(temporal_score)

    # =========================================================
    # INTER SUBJECT
    # =========================================================

    fingerprint = inter_subject_report.get(
        "fingerprint_score",
        0.5,
    )

    fingerprint = _clip01(fingerprint)

    consistency = 1.0 - fingerprint

    consistency = _clip01(consistency)

    # =========================================================
    # METADATA PRIOR
    # =========================================================

    metadata_prior = 0.5

    if metadata is not None:

        suitability = getattr(
            metadata,
            "hmm_suitability",
            "medium",
        )

        if suitability == "high":
            metadata_prior = 0.9

        elif suitability == "medium":
            metadata_prior = 0.6

        elif suitability == "low":
            metadata_prior = 0.3

    # =========================================================
    # FINAL SCORE
    # =========================================================

    final_score = (
        0.40 * gaussianity +
        0.35 * temporal_score +
        0.15 * consistency +
        0.10 * metadata_prior
    )

    final_score = _clip01(final_score)

    # =========================================================
    # LABEL
    # =========================================================

    if final_score >= 0.80:
        label = "excellent"

    elif final_score >= 0.65:
        label = "good"

    elif final_score >= 0.45:
        label = "acceptable"

    elif final_score >= 0.25:
        label = "poor"

    else:
        label = "very_poor"

    # =========================================================
    # WARNINGS
    # =========================================================

    warnings = []

    if gaussianity < 0.40:

        warnings.append(
            "Distribución poco gaussiana."
        )

    if temporal_score < 0.35:

        warnings.append(
            "Dinámica temporal demasiado ruidosa."
        )

    if consistency < 0.35:

        warnings.append(
            "Alta dependencia de identidad de sujeto."
        )

    # =========================================================
    # REPORT
    # =========================================================

    return {

        "final_score": float(final_score),

        "label": label,

        "subscores": {

            "gaussianity":
                float(gaussianity),

            "temporal_dynamics":
                float(temporal_score),

            "cross_subject_consistency":
                float(consistency),

            "subject_fingerprint_penalty":
                float(fingerprint),

            "metadata_prior":
                float(metadata_prior),
        },

        "weights": {

            "gaussianity": 0.40,
            "temporal_dynamics": 0.35,
            "cross_subject_consistency": 0.15,
            "metadata_prior": 0.10,
        },

        "warnings": warnings,
    }