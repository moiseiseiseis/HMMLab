# src/features/diagnostics/inter_subject.py

"""
Diagnósticos de variabilidad inter-sujeto para features EEG-HMM.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class InterSubjectReport:
    feature_name: str

    n_subjects: int

    intra_subject_variance: float
    inter_subject_variance: float

    subject_variability_ratio: float

    fingerprint_score: float

    interpretation: str


@dataclass
class SubjectDistanceReport:
    mean_distance: float
    std_distance: float

    min_distance: float
    max_distance: float

    interpretation: str


@dataclass
class SubjectCenteringEffect:
    variance_before: float
    variance_after: float

    reduction_ratio: float

    interpretation: str


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _safe_var(x: np.ndarray) -> float:

    if x.size <= 1:
        return 0.0

    return float(np.var(x))


def _flatten_feature(X: np.ndarray) -> np.ndarray:

    return X.reshape(-1)


# ---------------------------------------------------------------------
# Main variability analysis
# ---------------------------------------------------------------------

def analyze_inter_subject_variability(
    subject_feature_data: dict[str, np.ndarray],
    feature_name: str,
) -> InterSubjectReport:

    if len(subject_feature_data) < 2:
        raise ValueError(
            "Se requieren al menos 2 sujetos."
        )

    flattened = {}

    for subject, X in subject_feature_data.items():

        flattened[subject] = _flatten_feature(X)

    min_len = min(len(x) for x in flattened.values())

    for subject in flattened:
        flattened[subject] = flattened[subject][:min_len]

    intra_vars = []

    subject_means = []

    for subject, x in flattened.items():

        intra_vars.append(_safe_var(x))

        subject_means.append(np.mean(x))

    intra_subject_variance = float(np.mean(intra_vars))

    inter_subject_variance = float(
        np.var(subject_means)
    )

    ratio = (
        inter_subject_variance /
        (intra_subject_variance + 1e-12)
    )

    fingerprint_score = ratio / (1.0 + ratio)

    if fingerprint_score < 0.15:
        interpretation = "low_subject_signature"

    elif fingerprint_score < 0.35:
        interpretation = "moderate_subject_signature"

    elif fingerprint_score < 0.60:
        interpretation = "high_subject_signature"

    else:
        interpretation = "extreme_subject_signature"

    return InterSubjectReport(
        feature_name=feature_name,

        n_subjects=len(subject_feature_data),

        intra_subject_variance=intra_subject_variance,
        inter_subject_variance=inter_subject_variance,

        subject_variability_ratio=float(ratio),

        fingerprint_score=float(fingerprint_score),

        interpretation=interpretation,
    )


# ---------------------------------------------------------------------
# Subject separability
# ---------------------------------------------------------------------

def evaluate_subject_separability(
    subject_feature_data: dict[str, np.ndarray],
) -> SubjectDistanceReport:

    subjects = list(subject_feature_data.keys())

    if len(subjects) < 2:
        raise ValueError("Se requieren >=2 sujetos")

    centroids = []

    for subject in subjects:

        X = subject_feature_data[subject]

        centroid = np.mean(X, axis=0)

        centroid = np.ravel(centroid)

        centroids.append(centroid)

    centroids = np.asarray(centroids)

    distances = []

    for i in range(len(subjects)):

        for j in range(i + 1, len(subjects)):

            d = np.linalg.norm(
                centroids[i] - centroids[j]
            )

            distances.append(d)

    distances = np.asarray(distances)

    mean_d = float(np.mean(distances))
    std_d = float(np.std(distances))

    min_d = float(np.min(distances))
    max_d = float(np.max(distances))

    if mean_d < 0.5:
        interpretation = "subjects_overlap"

    elif mean_d < 2.0:
        interpretation = "moderate_subject_separation"

    else:
        interpretation = "strong_subject_separation"

    return SubjectDistanceReport(
        mean_distance=mean_d,
        std_distance=std_d,

        min_distance=min_d,
        max_distance=max_d,

        interpretation=interpretation,
    )


# ---------------------------------------------------------------------
# Subject centering impact
# ---------------------------------------------------------------------

def evaluate_subject_centering_impact(
    subject_feature_data: dict[str, np.ndarray],
) -> SubjectCenteringEffect:

    original = []
    centered = []

    for subject, X in subject_feature_data.items():

        original.append(X.reshape(-1))

        Xc = X - np.mean(X, axis=0, keepdims=True)

        centered.append(Xc.reshape(-1))

    original = np.concatenate(original)
    centered = np.concatenate(centered)

    var_before = float(np.var(original))
    var_after = float(np.var(centered))

    reduction = 1.0 - (
        var_after / (var_before + 1e-12)
    )

    if reduction < 0.10:
        interpretation = "minimal_subject_bias"

    elif reduction < 0.30:
        interpretation = "moderate_subject_bias"

    else:
        interpretation = "strong_subject_bias"

    return SubjectCenteringEffect(
        variance_before=var_before,
        variance_after=var_after,

        reduction_ratio=float(reduction),

        interpretation=interpretation,
    )


# ---------------------------------------------------------------------
# PUBLIC API WRAPPER
# ---------------------------------------------------------------------

def compute_inter_subject_diagnostics(
    X: np.ndarray,
    subject_slices: dict,
) -> dict:
    """
    Wrapper compatible con el pipeline principal.
    """

    subject_feature_data = {}

    for subject_id, sl in subject_slices.items():

        subject_feature_data[subject_id] = X[sl]

    variability = analyze_inter_subject_variability(
        subject_feature_data=subject_feature_data,
        feature_name="feature",
    )

    separability = evaluate_subject_separability(
        subject_feature_data
    )

    centering = evaluate_subject_centering_impact(
        subject_feature_data
    )

    return {

        "n_subjects":
            int(variability.n_subjects),

        "intra_subject_variance":
            float(variability.intra_subject_variance),

        "inter_subject_variance":
            float(variability.inter_subject_variance),

        "subject_variability_ratio":
            float(variability.subject_variability_ratio),

        "fingerprint_score":
            float(variability.fingerprint_score),

        "mean_inter_subject_distance":
            float(separability.mean_distance),

        "std_inter_subject_distance":
            float(separability.std_distance),

        "subject_centering_reduction":
            float(centering.reduction_ratio),

        "interpretation":
            variability.interpretation,
    }