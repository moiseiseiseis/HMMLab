#!/usr/bin/env python3
# scripts/features/compute_tde_embeddings.py
"""
Etapa 'embed' del pipeline TDE-HMM.

Qué hace:
  1. Lee los archivos *_epo.fif ya preprocesados (mismos que Featured-HMM)
  2. Para cada época/trial: construye el embedding TDE concatenando la señal
     con sus versiones desplazadas en el tiempo (lags 1..n_lags)
  3. Aplica StandardScaler + PCA global sobre todos los trials concatenados
  4. Guarda X_pca.npy, scaler.pkl, pca.pkl, lengths.npy y feature_manifest.json

Qué es el TDE (Time-Delay Embedding):
  Dado un vector de señal x(t) con C canales, el embedding de lag τ produce:
    [x(t), x(t-τ), x(t-2τ), ..., x(t-n_lags·τ)]
  con dimensión C × (n_lags + 1).

  Esto captura la covarianza temporal entre canales en diferentes momentos,
  permitiendo al HMM aprender estados definidos por patrones espacio-temporales
  en lugar de magnitudes de amplitud.

Uso:
  python scripts/features/compute_tde_embeddings.py --config configs/experiments/canonical/tde_k4_t7.yaml
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import mne
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

mne.set_log_level("ERROR")

# ── Localiza PROJECT_ROOT ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_path(p: str, root: Path) -> Path:
    return Path(str(root / p.replace("../../", "").replace("../", ""))).resolve()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tde_embedding(epoch_data: np.ndarray, n_lags: int, lag_step: int = 1) -> np.ndarray:
    """
    Construye el embedding TDE para una época.

    Args:
        epoch_data : (n_channels, n_times) — señal raw de un trial
        n_lags     : número de lags temporales (Oxford usa 7)
        lag_step   : paso entre lags en muestras (default 1 = cada muestra)

    Returns:
        embedding  : (n_times_valid, n_channels * (n_lags + 1))
                     n_times_valid = n_times - n_lags * lag_step

    Nota: el primer vector de tiempo válido es t = n_lags * lag_step,
    lo que evita índices negativos. Esto recorta ligeramente cada trial
    pero mantiene la alineación temporal.
    """
    n_channels, n_times = epoch_data.shape
    max_lag = n_lags * lag_step
    n_valid = n_times - max_lag

    if n_valid <= 0:
        raise ValueError(
            f"Trial demasiado corto para n_lags={n_lags}, lag_step={lag_step}. "
            f"n_times={n_times}, necesarios={max_lag + 1}"
        )

    # Stack: [x(t), x(t-lag), x(t-2*lag), ..., x(t-n_lags*lag)]
    parts = []
    for lag in range(0, n_lags + 1):
        offset = max_lag - lag * lag_step
        parts.append(epoch_data[:, offset:offset + n_valid])  # (C, n_valid)

    # Concatenar en eje de canales: (C*(n_lags+1), n_valid) → transponer → (n_valid, D)
    stacked = np.concatenate(parts, axis=0)  # (D, n_valid), D = C*(n_lags+1)
    return stacked.T  # (n_valid, D)


def process_subject(fif_path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee un archivo .fif y devuelve (embedding_matrix, lengths).

    Returns:
        X      : (total_samples, D) — embedding TDE de todos los trials
        lengths: (n_trials,)       — número de muestras por trial
    """
    n_lags   = cfg["tde"]["n_lags"]
    lag_step = cfg["tde"].get("lag_step", 1)

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    data   = epochs.get_data()  # (n_trials, n_channels, n_times)

    trial_embeddings = []
    trial_lengths    = []

    for trial_data in data:
        emb = build_tde_embedding(trial_data, n_lags=n_lags, lag_step=lag_step)
        trial_embeddings.append(emb)
        trial_lengths.append(emb.shape[0])

    X = np.vstack(trial_embeddings)
    lengths = np.array(trial_lengths, dtype=np.int32)
    return X, lengths


def main():
    parser = argparse.ArgumentParser(description="TDE Embedding + PCA")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Rutas ────────────────────────────────────────────────────────────────
    fif_dir    = _clean_path(cfg["paths"]["fif_dir"], PROJECT_ROOT)
    output_dir = _clean_path(cfg["paths"]["output_dir"], PROJECT_ROOT)
    exp_name   = cfg["experiment"]["name"]
    exp_dir    = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Parámetros TDE ───────────────────────────────────────────────────────
    n_lags     = cfg["tde"]["n_lags"]
    lag_step   = cfg["tde"].get("lag_step", 1)
    pca_var    = cfg["pipeline"]["pca"]["variance_retained"]

    print(f"\n{'='*60}")
    print(f"TDE Embedding — {exp_name}")
    print(f"  n_lags    : {n_lags}")
    print(f"  lag_step  : {lag_step}")
    print(f"  PCA var   : {pca_var*100:.0f}%")
    print(f"  fif_dir   : {fif_dir}")
    print(f"  output    : {exp_dir}")
    print(f"{'='*60}\n")

    # ── Encuentra archivos .fif ───────────────────────────────────────────────
    fif_files = sorted(fif_dir.glob("*_clean-epo.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No se encontraron archivos *_clean-epo.fif en {fif_dir}")

    print(f"Archivos encontrados: {len(fif_files)}")

    # ── Embedding por sujeto ─────────────────────────────────────────────────
    t0 = time.time()
    all_X       = []
    all_lengths = []
    manifest    = {}

    for i, fif_path in enumerate(fif_files, 1):
        print(f"  [{i:3d}/{len(fif_files)}] {fif_path.name}", end=" ", flush=True)
        X_sub, lengths_sub = process_subject(fif_path, cfg)
        all_X.append(X_sub)
        all_lengths.append(lengths_sub)
        manifest[fif_path.name] = _md5(fif_path)
        print(f"-> {X_sub.shape[0]:,} muestras, D={X_sub.shape[1]}")

    X_all   = np.vstack(all_X)
    lengths = np.concatenate(all_lengths).astype(np.int32)
    D_raw   = X_all.shape[1]

    print(f"\nEmbedding total: {X_all.shape[0]:,} × {D_raw}")
    print(f"Trials totales : {len(lengths)} | longitud modal: {np.bincount(lengths).argmax()}")

    # ── StandardScaler ───────────────────────────────────────────────────────
    print("\nAjustando StandardScaler...")
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ── PCA ──────────────────────────────────────────────────────────────────
    print(f"Ajustando PCA (varianza retenida={pca_var*100:.0f}%)...")
    pca = PCA(n_components=pca_var, svd_solver="full", random_state=cfg["experiment"]["seed"])
    X_pca = pca.fit_transform(X_scaled)
    n_pcs = X_pca.shape[1]

    elapsed = time.time() - t0
    print(f"PCA: {D_raw}D -> {n_pcs} PCs ({pca.explained_variance_ratio_.sum()*100:.1f}% varianza)")
    print(f"Tiempo total embedding: {elapsed:.1f}s")

    # ── Guarda outputs ───────────────────────────────────────────────────────
    print(f"\nGuardando en {exp_dir}...")

    np.save(exp_dir / "X_pca.npy",   X_pca)
    np.save(exp_dir / "X_scaled.npy", X_scaled)  # para topomapas
    np.save(exp_dir / "lengths.npy", lengths)
    joblib.dump(scaler, exp_dir / "scaler.pkl")
    joblib.dump(pca,    exp_dir / "pca.pkl")

    # Manifest MD5
    manifest_data = {
        "fif_files": manifest,
        "n_files":   len(fif_files),
        "n_samples": int(X_all.shape[0]),
        "D_raw":     int(D_raw),
        "n_pcs":     int(n_pcs),
        "n_lags":    n_lags,
        "lag_step":  lag_step,
        "pca_var_target":   pca_var,
        "pca_var_achieved": float(pca.explained_variance_ratio_.sum()),
    }
    (exp_dir / "feature_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    print(f"OK Embedding completado:")
    print(f"   X_pca.npy   : {X_pca.shape}")
    print(f"   lengths.npy : {lengths.shape} | suma={lengths.sum():,}")
    print(f"   n_pcs       : {n_pcs}")
    print(f"   manifest     : feature_manifest.json")


if __name__ == "__main__":
    main()