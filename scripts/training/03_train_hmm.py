# scripts/training/03_train_hmm.py
import os
import sys
import argparse
import json
import yaml
import numpy as np
import joblib
from hmmlearn import hmm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT, clean_path

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento estocastico de HMM")
    parser.add_argument('--config', type=str, required=True, help="Ruta al YAML del experimento")
    args = parser.parse_args()

    # 1. Cargar Configuracion
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg['experiment']['name']
    seed = cfg['experiment']['seed']
    k_states = cfg['pipeline']['hmm']['k_states']
    cov_type = cfg['pipeline']['hmm']['covariance_type']
    n_iter = cfg['pipeline']['hmm']['n_iter']
    tol = cfg['pipeline']['hmm'].get('tol', 0.001)
    
    # 2. RUTA BLINDADA (Aqui estaba el error antes)
    out_dir_base = clean_path(cfg['paths']['output_dir'])
    exp_output_dir = os.path.join(out_dir_base, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    print(f"\n=== ENTRENAMIENTO HMM: {exp_name} ===")
    print(f"Estados (K): {k_states} | Covarianza: {cov_type} | Semilla: {seed}")
    print(f"Directorio de trabajo: {exp_output_dir}") # Esto ahora imprimira C:\Proyectos\...
    
    # 3. Cargar Datos
    pca_matrix_path = os.path.join(exp_output_dir, 'X_pca.npy')
    lengths_path = os.path.join(exp_output_dir, 'lengths.npy')
    
    try:
        print("Cargando matriz PCA y longitudes...")
        X_pca = np.load(pca_matrix_path)
        lengths = np.load(lengths_path)
        lengths = [int(l) for l in lengths] 
    except FileNotFoundError:
        print(f"\nERROR ERROR FATAL: No se encontro X_pca.npy o lengths.npy en:\n{exp_output_dir}")
        return

    print(f"Datos listos: {X_pca.shape[0]} ventanas, {X_pca.shape[1]} componentes PCA.")
    
    # 4. Entrenamiento
    print(f"\nIniciando entrenamiento HMM...")
    model = hmm.GaussianHMM(
        n_components=k_states, covariance_type=cov_type,
        n_iter=n_iter, tol=tol, random_state=seed, verbose=True
    )

    model.fit(X_pca, lengths)

    history = model.monitor_.history
    convergence_log = {
        "converged": bool(model.monitor_.converged),
        "n_iter_run": int(model.monitor_.iter),
        "delta_final": float(history[-1] - history[-2]) if len(history) >= 2 else None,
        "log_likelihoods": [float(v) for v in history],
    }
    with open(os.path.join(exp_output_dir, 'convergence_log.json'), 'w', encoding='utf-8') as cj:
        json.dump(convergence_log, cj, indent=2)

    if model.monitor_.converged:
        print(f"\nOK El modelo convergio magistralmente en el paso {model.monitor_.iter}!")
    else:
        print(f"\nWARN Advertencia: El modelo NO convergio despues de {n_iter} iteraciones.")
        
    # 5. Viterbi y Guardado
    print("Calculando caminos Viterbi...")
    _, viterbi_paths = model.decode(X_pca, lengths)
    
    model_path = os.path.join(exp_output_dir, f'hmm_model_k{k_states}.pkl')
    viterbi_path = os.path.join(exp_output_dir, f'viterbi_paths_k{k_states}.npy')
    
    joblib.dump(model, model_path)
    np.save(viterbi_path, viterbi_paths)
    
    print("\n================ GUARDADO EXITOSO ================")
    print(f"[dir] Directorio: {exp_output_dir}")
    print(f"[file] Modelo guardado como: hmm_model_k{k_states}.pkl")
    print("==================================================")

if __name__ == "__main__":
    main()