# scripts/features/02_fit_pca.py
import os
import sys
import argparse
import yaml
import glob
import hashlib
import json
import datetime
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT, clean_path

def main():
    parser = argparse.ArgumentParser(description="Ajustar PCA sobre caracteristicas")
    parser.add_argument('--config', type=str, required=True, help="Ruta al YAML del experimento")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg['experiment']['name']
    variance = cfg['pipeline']['pca']['variance_retained']
    
    # 1. Rutas blindadas y limpias
    features_dir = clean_path(cfg['paths']['features_dir'])
    out_dir_base = clean_path(cfg['paths']['output_dir'])
    
    exp_output_dir = os.path.join(out_dir_base, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    print(f"\n=== PCA PARA EXPERIMENTO: {exp_name} ===")
    
    feature_files = sorted(glob.glob(os.path.join(features_dir, '*_features.npy')))
    length_files = sorted(glob.glob(os.path.join(features_dir, '*_lengths.npy')))

    if not feature_files:
        print(f"ERROR ERROR: No se encontraron archivos *_features.npy en {features_dir}")
        return

    # manifest: MD5 + orden exacto de archivos
    def _md5(path):
        h = hashlib.md5()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "files": [
            {"path": os.path.relpath(f, PROJECT_ROOT), "md5": _md5(f)}
            for f in feature_files
        ],
    }
    with open(os.path.join(exp_output_dir, 'feature_manifest.json'), 'w', encoding='utf-8') as fj:
        json.dump(manifest, fj, indent=2)

    np.save(
        os.path.join(exp_output_dir, 'file_order.npy'),
        np.array(feature_files),
    )

    # 2. Carga y concatenacion
    X_all, lengths_all = [], []
    for f, l in zip(feature_files, length_files):
        X_all.append(np.load(f))
        lengths_all.extend(np.load(l))
        
    X = np.vstack(X_all)
    print(f"Matriz Global concatenada: {X.shape}. Estandarizando...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Aplicando PCA (Reteniendo {variance*100}% de varianza)...")
    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Dimension final tras PCA: {X_pca.shape[1]} componentes.")
    
    # 3. Guardado estricto en la carpeta del experimento
    np.save(os.path.join(exp_output_dir, 'X_pca.npy'), X_pca)
    np.save(os.path.join(exp_output_dir, 'lengths.npy'), np.array(lengths_all))
    joblib.dump(pca, os.path.join(exp_output_dir, 'pca_model.pkl'))
    joblib.dump(scaler, os.path.join(exp_output_dir, 'scaler.pkl'))
    
    print(f"OK Archivos PCA guardados exitosamente en:\n{exp_output_dir}")

if __name__ == "__main__":
    main()