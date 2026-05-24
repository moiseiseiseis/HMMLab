"""
Verificación de determinismo del pipeline de extracción de features.

Compara X_pca.npy del experimento canonical_k4_full_95d contra una
re-ejecución completa (01_extract_features + 02_fit_pca) en _verify/.

NO modifica ningún experimento existente.
"""

import os
import sys
import io
import shutil
import subprocess
import yaml
import numpy as np
from pathlib import Path

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows (cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.paths import PROJECT_ROOT


# ============================================================
# RUTAS
# ============================================================

ORIGINAL_YAML  = PROJECT_ROOT / "configs/experiments/canonical/canonical_k4_full_95d.yaml"
ORIGINAL_XPCA  = PROJECT_ROOT / "outputs/experiments/canonical/canonical_k4_full_95d/X_pca.npy"

VERIFY_FEAT_DIR  = PROJECT_ROOT / "data/interim/features/_verify"
VERIFY_OUT_DIR   = PROJECT_ROOT / "outputs/experiments/_verify"
TEMP_YAML_PATH   = PROJECT_ROOT / "configs/experiments/_verify_temp.yaml"

SCRIPT_EXTRACT   = PROJECT_ROOT / "scripts/features/01_extract_features.py"
SCRIPT_PCA       = PROJECT_ROOT / "scripts/features/02_fit_pca.py"


# ============================================================
# HELPERS
# ============================================================

def create_temp_yaml() -> Path:
    """Copia el YAML original redirigiendo paths a _verify/."""
    with open(ORIGINAL_YAML, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg['paths']['features_dir'] = "data/interim/features/_verify"
    cfg['paths']['output_dir']   = "outputs/experiments/_verify"

    with open(TEMP_YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"[+] YAML temporal creado: {TEMP_YAML_PATH}")
    return TEMP_YAML_PATH


def run_script(script: Path, yaml_path: Path) -> int:
    """Ejecuta un script Python con --config y devuelve el returncode."""
    cmd = [sys.executable, str(script), "--config", str(yaml_path)]
    print(f"\n[>] Ejecutando: {' '.join(cmd)}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return result.returncode


def compare_xpca(orig: Path, new: Path) -> bool:
    """Carga ambos arrays y los compara con np.allclose(atol=1e-6)."""
    A = np.load(orig)
    B = np.load(new)

    print(f"\n=== COMPARACIÓN DE X_PCA ===")
    print(f"Original : {A.shape}  dtype={A.dtype}")
    print(f"Nuevo    : {B.shape}  dtype={B.dtype}")

    if A.shape != B.shape:
        print(f"[!] FORMAS DISTINTAS: {A.shape} vs {B.shape}")
        return False

    equal = np.allclose(A, B, atol=1e-6)
    print(f"\nnp.allclose(atol=1e-6) -> {equal}")

    if not equal:
        diff = np.abs(A - B)
        n_diff = np.sum(diff > 1e-6)
        print(f"Elementos distintos (|diff| > 1e-6): {n_diff} / {A.size}  "
              f"({100*n_diff/A.size:.3f}%)")
        print(f"Max diff  : {diff.max():.2e}")
        print(f"Mean diff : {diff.mean():.2e}")

        # Primeras 5 posiciones con diferencia
        idx = np.argwhere(diff > 1e-6)[:5]
        print("\nPrimeras diferencias (fila, col):")
        for r, c in idx:
            print(f"  [{r:6d}, {c:3d}]  orig={A[r,c]:.8f}  nuevo={B[r,c]:.8f}  "
                  f"delta={diff[r,c]:.2e}")

    return equal


def cleanup():
    """Borra los directorios y YAML temporales."""
    for path in [VERIFY_FEAT_DIR, VERIFY_OUT_DIR, TEMP_YAML_PATH]:
        if isinstance(path, Path) and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print("[+] Directorios temporales eliminados.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  VERIFICACIÓN DE DETERMINISMO — canonical_k4_full_95d")
    print("=" * 60)

    # Paso 0: Verificar que el X_pca original existe
    if not ORIGINAL_XPCA.exists():
        print(f"[ERROR] No se encontró el archivo original:\n  {ORIGINAL_XPCA}")
        sys.exit(1)
    print(f"[+] X_pca original localizado: {ORIGINAL_XPCA}")
    X_orig_check = np.load(ORIGINAL_XPCA)
    print(f"    Shape={X_orig_check.shape}  dtype={X_orig_check.dtype}")
    del X_orig_check

    # Paso 1: YAML temporal
    temp_yaml = create_temp_yaml()

    try:
        # Paso 2: Extracción de features
        print("\n--- STAGE 1: Extracción de features ---")
        rc = run_script(SCRIPT_EXTRACT, temp_yaml)
        if rc != 0:
            print(f"[ERROR] 01_extract_features.py terminó con código {rc}")
            sys.exit(rc)

        # Paso 3: PCA
        print("\n--- STAGE 2: PCA ---")
        rc = run_script(SCRIPT_PCA, temp_yaml)
        if rc != 0:
            print(f"[ERROR] 02_fit_pca.py terminó con código {rc}")
            sys.exit(rc)

        # Paso 4: Comparación
        # El experimento name sigue siendo canonical_k4_full_95d
        exp_name = yaml.safe_load(open(ORIGINAL_YAML))['experiment']['name']
        new_xpca = VERIFY_OUT_DIR / exp_name / "X_pca.npy"

        if not new_xpca.exists():
            print(f"[ERROR] No se generó el nuevo X_pca.npy en:\n  {new_xpca}")
            sys.exit(1)

        result = compare_xpca(ORIGINAL_XPCA, new_xpca)

        print("\n" + "=" * 60)
        print(f"  RESULTADO FINAL: pipeline {'ES' if result else 'NO ES'} determinista")
        print("=" * 60)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
