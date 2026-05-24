#!/usr/bin/env python3
# scripts/training/train_tde_hmm.py
"""
Etapa 'hmm' del pipeline TDE-HMM.

Entrena un GaussianHMM sobre el espacio PCA del embedding TDE.
Idéntico en lógica al Featured-HMM — la diferencia está en la representación
de entrada (TDE en lugar de features manuales), no en el modelo.

Uso:
  python scripts/training/train_tde_hmm.py --config configs/experiments/canonical/tde_k4_t7.yaml
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM

# ── Localiza PROJECT_ROOT ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_path(p: str, root: Path) -> Path:
    return Path(str(root / p.replace("../../", "").replace("../", ""))).resolve()


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento HMM — pipeline TDE")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Parámetros ────────────────────────────────────────────────────────────
    exp_name   = cfg["experiment"]["name"]
    seed       = cfg["experiment"]["seed"]
    output_dir = _clean_path(cfg["paths"]["output_dir"], PROJECT_ROOT)
    exp_dir    = output_dir / exp_name

    K          = cfg["pipeline"]["hmm"]["k_states"]
    cov_type   = cfg["pipeline"]["hmm"]["covariance_type"]
    n_iter     = cfg["pipeline"]["hmm"].get("n_iter", 500)
    tol        = cfg["pipeline"]["hmm"].get("tol", 0.001)

    print(f"\n{'='*60}")
    print(f"TDE-HMM Training — {exp_name}")
    print(f"  K={K} | cov={cov_type} | n_iter={n_iter} | tol={tol} | seed={seed}")
    print(f"{'='*60}\n")

    # ── Carga datos ───────────────────────────────────────────────────────────
    X_pca   = np.load(exp_dir / "X_pca.npy")
    lengths = np.load(exp_dir / "lengths.npy")

    print(f"X_pca   : {X_pca.shape}")
    print(f"lengths : {lengths.shape} | suma={lengths.sum():,} | modal={np.bincount(lengths).argmax()}")

    # Verificación de integridad
    assert lengths.sum() == X_pca.shape[0], (
        f"Mismatch: lengths.sum()={lengths.sum()} != X_pca.shape[0]={X_pca.shape[0]}"
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print(f"\nEntrenando GaussianHMM K={K}...")
    t0 = time.time()

    model = GaussianHMM(
        n_components=K,
        covariance_type=cov_type,
        n_iter=n_iter,
        tol=tol,
        random_state=seed,
        verbose=False,
    )
    model.fit(X_pca, lengths=lengths)

    elapsed = time.time() - t0
    converged = model.monitor_.converged

    print(f"\nConvergencia : {'OK SI' if converged else 'WARN NO (maximo de iteraciones alcanzado)'}")
    print(f"Iteraciones  : {len(model.monitor_.history)}")
    print(f"Log-likelihood final: {model.monitor_.history[-1]:.4f}")
    print(f"Tiempo       : {elapsed:.1f}s")

    # ── Viterbi ───────────────────────────────────────────────────────────────
    print("\nCalculando caminos Viterbi...")
    viterbi = model.predict(X_pca, lengths=lengths)

    # ── Métricas rápidas ─────────────────────────────────────────────────────
    transmat = model.transmat_
    print("\nMétricas por estado:")
    print(f"  {'Estado':<8} {'FO':>8} {'Self-trans':>12} {'Dwell(ms)':>12} {'Flag'}")
    print(f"  {'-'*55}")
    step_ms = float(cfg.get("tde", {}).get("step_ms", 4.0))  # default 1/sfreq_after_pca
    for s in range(K):
        fo     = np.mean(viterbi == s)
        self_t = transmat[s, s]
        dwell  = step_ms / max(1 - self_t, 1e-9)
        flag   = "WARN ATRACTOR" if self_t > 0.97 or dwell > 5000 else "OK"
        print(f"  S{s}      {fo:>8.4f} {self_t:>12.4f} {dwell:>12.1f}   {flag}")

    # ── Guarda outputs ────────────────────────────────────────────────────────
    print(f"\nGuardando en {exp_dir}...")

    joblib.dump(model, exp_dir / f"hmm_model_k{K}.pkl")
    np.save(exp_dir / f"viterbi_paths_k{K}.npy", viterbi)

    run_info = {
        "experiment": exp_name,
        "pipeline_type": "tde",
        "status": "completed",
        "converged": bool(converged),
        "n_iter_run": len(model.monitor_.history),
        "log_likelihood_final": float(model.monitor_.history[-1]),
        "elapsed_seconds": round(elapsed, 1),
        "K": K,
        "covariance_type": cov_type,
        "n_pcs": int(X_pca.shape[1]),
        "n_samples": int(X_pca.shape[0]),
        "n_trials": int(len(lengths)),
        "seed": seed,
        "ll_history": [float(v) for v in model.monitor_.history],
    }
    (exp_dir / "run.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    print(f"OK Entrenamiento completado")
    print(f"   hmm_model_k{K}.pkl")
    print(f"   viterbi_paths_k{K}.npy")
    print(f"   run.json")


if __name__ == "__main__":
    main()