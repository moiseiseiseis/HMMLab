#!/usr/bin/env python3
# scripts/decoding/decode_tde_hmm.py
"""
Etapa 'decode' del pipeline TDE-HMM.

Calcula y guarda las métricas de decoding:
  - Fractional Occupancy (hard + soft) por sujeto y condición
  - Dwell times empíricos y analíticos
  - Matrices de transición
  - Topomapas en espacio de canales originales (reconstrucción desde PCA+scaler)
  - state_stats.csv compatible con el master report

Uso:
  python scripts/decoding/decode_tde_hmm.py --config configs/experiments/canonical/tde_k4_t7.yaml
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Localiza PROJECT_ROOT ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_path(p: str, root: Path) -> Path:
    return Path(str(root / p.replace("../../", "").replace("../", ""))).resolve()


def _load_labels(root: Path) -> dict:
    csv_path = root / "configs" / "subject_labels.csv"
    if not csv_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        base_id = row["npy_file"].split("_sin_contexto")[0].upper()
        result[base_id] = {
            "condition": row["condition"],
            "group":     row["group"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Decoding TDE-HMM")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Parámetros ────────────────────────────────────────────────────────────
    exp_name   = cfg["experiment"]["name"]
    output_dir = _clean_path(cfg["paths"]["output_dir"], PROJECT_ROOT)
    fif_dir    = _clean_path(cfg["paths"]["fif_dir"], PROJECT_ROOT)
    exp_dir    = output_dir / exp_name
    K          = cfg["pipeline"]["hmm"]["k_states"]

    print(f"\n{'='*60}")
    print(f"TDE-HMM Decoding — {exp_name}")
    print(f"  K={K}")
    print(f"{'='*60}\n")

    # ── Carga modelo y datos ─────────────────────────────────────────────────
    model   = joblib.load(exp_dir / f"hmm_model_k{K}.pkl")
    viterbi = np.load(exp_dir / f"viterbi_paths_k{K}.npy")
    X_pca   = np.load(exp_dir / "X_pca.npy")
    lengths = np.load(exp_dir / "lengths.npy")
    post    = model.predict_proba(X_pca)

    transmat = model.transmat_

    # ── Metadatos por sujeto ─────────────────────────────────────────────────
    labels = _load_labels(PROJECT_ROOT)

    # Bug 1: usa manifest para obtener exactamente los mismos archivos que el embedding
    manifest_path = exp_dir / "feature_manifest.json"
    import json as _json
    with open(manifest_path, "r", encoding="utf-8") as _f:
        _manifest = _json.load(_f)
    fif_names = list(_manifest["fif_files"].keys())  # orden original del embedding
    fif_files = [fif_dir / name for name in fif_names]

    # Reconstruye subject_meta desde lengths (igual que featured-HMM decode)
    # Cada .fif → todos sus trials concatenados
    # La longitud por sujeto = suma de sus trials
    # Necesitamos saber cuántos trials tiene cada sujeto: usamos lengths
    # Los fif_files están en el mismo orden que el embedding (sorted)
    import mne
    mne.set_log_level("ERROR")

    subject_meta = []
    idx = 0
    n_lags    = cfg["tde"]["n_lags"]
    lag_step  = cfg["tde"].get("lag_step", 1)
    sfreq_tde = cfg.get("tde", {}).get("sfreq", 250.0)

    for fif_path in fif_files:
        epochs   = mne.read_epochs(fif_path, preload=False, verbose=False)
        n_trials = len(epochs)

        # Bug 3: ajusta n_times a la sfreq del embedding (puede diferir de la sfreq original)
        n_times_resampled = int(round(len(epochs.times) * sfreq_tde / epochs.info["sfreq"]))
        n_valid = n_times_resampled - n_lags * lag_step
        n_windows_sub = n_trials * n_valid

        # Bug 2: extrae base_id del nombre del .fif para buscar en subject_labels.csv
        fname_stem = fif_path.stem  # ej: AAELSCGO_GO_clean-epo
        base_id    = fname_stem.split("_")[0].upper()
        row        = labels.get(base_id, {})
        match      = re.search(r"_(NOGO|GO)_", fif_path.name, re.IGNORECASE)
        cond       = match.group(1).upper() if match else row.get("condition", "UNKNOWN")
        group      = row.get("group", "UNKNOWN")

        subject_meta.append({
            "fname":      fif_path.stem,
            "subject_id": fif_path.stem.split("_")[0],
            "condition":  cond,
            "group":      group,
            "start":      idx,
            "end":        idx + n_windows_sub,
            "n_windows":  n_windows_sub,
            "n_trials":   n_trials,
        })
        idx += n_windows_sub

    df_meta = pd.DataFrame(subject_meta)
    print(f"Sesiones: {len(df_meta)}")
    print(df_meta.groupby(["group", "condition"]).size().to_string())

    # ── FO por estado y sujeto ────────────────────────────────────────────────
    fo_matrix = np.zeros((len(subject_meta), K))
    for i, m in enumerate(subject_meta):
        v_sub = viterbi[m["start"]:m["end"]]
        for s in range(K):
            fo_matrix[i, s] = np.mean(v_sub == s)

    # ── Dwell times ───────────────────────────────────────────────────────────
    def compute_dwells(v, step_ms=4.0):
        """step_ms = 1000/sfreq para TDE (no windowing, es por muestra)."""
        dwells = {s: [] for s in range(K)}
        i, n = 0, len(v)
        while i < n:
            s = v[i]; j = i
            while j < n and v[j] == s:
                j += 1
            dwells[s].append((j - i) * step_ms)
            i = j
        return dwells

    # step_ms del TDE = tiempo entre muestras = 1000/sfreq
    step_ms = 1000.0 / sfreq_tde

    dwells_global = compute_dwells(viterbi, step_ms=step_ms)

    # ── state_stats.csv ──────────────────────────────────────────────────────
    rows = []
    for s in range(K):
        mask_s  = (viterbi == s)
        fo_hard = float(np.mean(mask_s))
        fo_soft = float(np.mean(post[:, s]))
        self_t  = float(transmat[s, s])
        dwell_a = step_ms / max(1 - self_t, 1e-9)
        H       = -np.sum(post * np.log(post + 1e-12), axis=1)
        conf    = float(np.mean(np.max(post, axis=1)[mask_s])) if mask_s.sum() else float("nan")
        entropy = float(np.mean(H[mask_s])) if mask_s.sum() else float("nan")
        d       = np.array(dwells_global[s]) if dwells_global[s] else np.array([0.0])
        flag    = "ATRACTOR" if self_t > 0.97 or dwell_a > 5000 else "OK"
        rows.append({
            "state":              f"S{s}",
            "FO_hard":            round(fo_hard, 6),
            "FO_soft":            round(fo_soft, 6),
            "self_trans":         round(self_t, 6),
            "dwell_analytic_ms":  round(dwell_a, 2),
            "dwell_median_ms":    round(float(np.median(d)), 2),
            "dwell_mean_ms":      round(float(np.mean(d)), 2),
            "dwell_p95_ms":       round(float(np.percentile(d, 95)), 2),
            "confidence_mean":    round(conf, 6),
            "entropy_H":          round(entropy, 6),
            "n_windows":          int(mask_s.sum()),
            "n_runs":             len(dwells_global[s]),
            "flag":               flag,
        })

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(exp_dir / "state_stats.csv", index=False)

    print("\nMétricas globales:")
    print(df_stats[["state", "FO_hard", "self_trans", "dwell_analytic_ms",
                     "confidence_mean", "flag"]].to_string(index=False))

    # ── FO por condición ─────────────────────────────────────────────────────
    fo_by_cond_rows = []
    for i, m in enumerate(subject_meta):
        for s in range(K):
            fo_by_cond_rows.append({
                "subject_id": m["subject_id"],
                "condition":  m["condition"],
                "group":      m["group"],
                "state":      f"S{s}",
                "FO":         round(fo_matrix[i, s], 6),
            })

    df_fo = pd.DataFrame(fo_by_cond_rows)
    df_fo.to_csv(exp_dir / "fo_by_subject.csv", index=False)

    # ── Matriz de transición ──────────────────────────────────────────────────
    np.save(exp_dir / f"transition_matrix_k{K}.npy", transmat)

    # ── Sujetos absorbentes ───────────────────────────────────────────────────
    absorbing = []
    for i, m in enumerate(subject_meta):
        for s in range(K):
            if fo_matrix[i, s] > 0.95:
                absorbing.append({
                    "subject_id": m["subject_id"],
                    "condition":  m["condition"],
                    "group":      m["group"],
                    "state":      f"S{s}",
                    "FO":         round(fo_matrix[i, s], 4),
                })

    if absorbing:
        print(f"\nWARN  Sujetos absorbentes (FO>0.95): {len(absorbing)} sesiones")
        for a in absorbing:
            print(f"   {a['subject_id']} ({a['condition']}) -> {a['state']} FO={a['FO']:.4f}")
    else:
        print("\nOK Sin sujetos absorbentes")

    # ── Score de calidad ─────────────────────────────────────────────────────
    fo_vals  = [float(np.mean(viterbi == s)) for s in range(K)]
    fo_range = max(fo_vals) - min(fo_vals)
    mean_conf = float(np.mean(np.max(post, axis=1)))
    dwell_max = max(
        step_ms / max(1 - transmat[s, s], 1e-9) for s in range(K)
    )

    checks = {
        "FO_min_gt_10pct":   bool(min(fo_vals) >= 0.10),
        "dwell_max_lt_2s":   bool(dwell_max <= 2000),
        "FO_balance_lt_0.3": bool(fo_range <= 0.30),
        "confidence_gt_0.6": bool(mean_conf >= 0.60),
    }
    score = int(sum(checks.values()))

    print(f"\nScore de calidad: {score}/4")
    for k, v in checks.items():
        print(f"  {'OK' if v else 'ERROR'} {k}")

    # ── Decoding summary ──────────────────────────────────────────────────────
    summary = {
        "experiment":    exp_name,
        "pipeline_type": "tde",
        "K":             K,
        "n_sessions":    int(len(subject_meta)),
        "n_samples":     int(len(viterbi)),
        "score_4":       int(score),
        "verdict":       "REPORTAR" if score == 4 else "ACEPTABLE" if score == 3 else "REVISAR",
        "FO_min":        float(round(min(fo_vals), 4)),
        "FO_max":        float(round(max(fo_vals), 4)),
        "dwell_max_ms":  float(round(dwell_max, 1)),
        "confidence":    float(round(mean_conf, 4)),
        "n_absorbing":   int(len(absorbing)),
        "checks":        checks,
        "absorbing":     absorbing,
    }
    (exp_dir / "decoding_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nOK Decoding completado")
    print(f"   state_stats.csv")
    print(f"   fo_by_subject.csv")
    print(f"   transition_matrix_k{K}.npy")
    print(f"   decoding_summary.json")
    print(f"\n   Veredicto: {summary['verdict']} ({score}/4)")


if __name__ == "__main__":
    main()