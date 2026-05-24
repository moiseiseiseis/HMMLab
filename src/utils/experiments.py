# ============================================
# src/utils/experiment.py
# ============================================

def build_experiment_id(
    paradigm,
    model,
    n_states,
    pca=None,
    lags=None,
    reference=None,
    downsample=None
):

    parts = [
        paradigm,
        model,
        f"K{n_states}"
    ]

    if pca is not None:
        parts.append(f"PCA{pca}")

    if lags is not None:
        parts.append(f"L{lags}")

    if reference is not None:
        parts.append(reference.upper())

    if downsample is not None:
        parts.append(f"DS{downsample}")

    return "_".join(parts)