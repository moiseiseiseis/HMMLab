# src/features/diagnostics/report.py
"""
Generación de reportes para validación de features EEG-HMM.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def save_json_report(
    report: dict,
    output_path: str | Path,
) -> None:
    """
    Guarda reporte JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------

def dataclass_to_dict(obj):
    """
    Convierte dataclasses recursivamente.
    """

    if isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]

    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: dataclass_to_dict(v)
            for k, v in asdict(obj).items()
        }

    return obj


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

def generate_markdown_report(
    *,
    feature_name: str,

    metadata: dict,

    distribution_result=None,
    smoothness_result=None,
    stationarity_result=None,
    redundancy_result=None,
    inter_subject_result=None,
    hmm_score_result=None,
) -> str:
    """
    Genera reporte markdown legible.
    """

    lines = []

    lines.append(f"# Feature Diagnostic Report")
    lines.append("")
    lines.append(f"## {feature_name}")
    lines.append("")

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------
    lines.append("## Metadata")
    lines.append("")

    for k, v in metadata.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("")

    # -------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------
    if distribution_result is not None:

        lines.append("## Distribution")
        lines.append("")

        d = dataclass_to_dict(distribution_result)

        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")

        lines.append("")

    # -------------------------------------------------------------
    # Smoothness
    # -------------------------------------------------------------
    if smoothness_result is not None:

        lines.append("## Temporal Smoothness")
        lines.append("")

        d = dataclass_to_dict(smoothness_result)

        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")

        lines.append("")

    # -------------------------------------------------------------
    # Stationarity
    # -------------------------------------------------------------
    if stationarity_result is not None:

        lines.append("## Stationarity")
        lines.append("")

        d = dataclass_to_dict(stationarity_result)

        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")

        lines.append("")

    # -------------------------------------------------------------
    # Redundancy
    # -------------------------------------------------------------
    if redundancy_result is not None:

        lines.append("## Redundancy")
        lines.append("")

        d = dataclass_to_dict(redundancy_result)

        for k, v in d.items():

            if k == "redundant_pairs":

                lines.append("- **redundant_pairs**:")

                for pair in v:

                    lines.append(
                        f"  - {pair['feature_a']} ↔ "
                        f"{pair['feature_b']} "
                        f"(corr={pair['correlation']:.3f})"
                    )

            else:
                lines.append(f"- **{k}**: {v}")

        lines.append("")

    # -------------------------------------------------------------
    # Inter-subject
    # -------------------------------------------------------------
    if inter_subject_result is not None:

        lines.append("## Inter-Subject Variability")
        lines.append("")

        d = dataclass_to_dict(inter_subject_result)

        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")

        lines.append("")

    # -------------------------------------------------------------
    # HMM Score
    # -------------------------------------------------------------
    if hmm_score_result is not None:

        lines.append("## HMM Compatibility")
        lines.append("")

        d = dataclass_to_dict(hmm_score_result)

        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Save markdown
# ---------------------------------------------------------------------

def save_markdown_report(
    markdown_text: str,
    output_path: str | Path,
) -> None:
    """
    Guarda reporte markdown.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)