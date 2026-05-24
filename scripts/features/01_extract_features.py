# scripts/features/01_extract_features.py

import os
import sys
import glob
import yaml
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT, clean_path

import mne
import numpy as np

from tqdm import tqdm

# ============================================================
# NUEVO SISTEMA MODULAR
# ============================================================

# IMPORTANTE:
# Esto auto-registra TODAS las features vía __init__.py
import src.features

from src.features.extractor_engine import ModularFeatureExtractor
from src.features.registry import REGISTRY


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Extracción modular de features EEG-HMM"
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Ruta al YAML del experimento"
    )

    args = parser.parse_args()

    # ========================================================
    # CARGA DE CONFIGURACIONES
    # ========================================================

    with open(args.config, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    with open(
        os.path.join(
            PROJECT_ROOT,
            'configs/preprocessing/task_feature_hmm.yaml'
        ),
        'r',
        encoding='utf-8'
    ) as f:

        pre_cfg = yaml.safe_load(f)['preprocessing']

    # ========================================================
    # PARÁMETROS
    # ========================================================

    fs = pre_cfg['resampling']['target']

    win_cfg = exp_cfg['windowing']
    win_samples = int(win_cfg['window_ms'] * fs / 1000)
    step_samples = int(win_cfg['step_ms'] * fs / 1000)
    edge_trim = win_cfg.get('edge_trim', 0)

    feature_flags = exp_cfg.get('features', {})

    # ========================================================
    # NUEVO EXTRACTOR MODULAR
    # ========================================================

    extractor = ModularFeatureExtractor(feature_flags)

    # ========================================================
    # DEBUG / LOGGING
    # ========================================================

    print("\n")
    REGISTRY.print_summary()

    extractor.summary()

    # ========================================================
    # RUTAS
    # ========================================================

    input_dir = os.path.join(
        PROJECT_ROOT,
        'data/interim/preprocessed/task/'
    )

    out_dir = clean_path(
        exp_cfg['paths']['features_dir'],
    )

    os.makedirs(out_dir, exist_ok=True)

    # ========================================================
    # INPUT FILES
    # ========================================================

    fif_files = sorted(
        glob.glob(
            os.path.join(input_dir, '*_epo.fif')
        )
    )

    # ========================================================
    # INFO INICIAL
    # ========================================================

    print("=== INICIANDO EXTRACCIÓN MODULAR DE FEATURES ===")
    print(f"Sujetos encontrados: {len(fif_files)}")
    print(f"Directorio de salida: {out_dir}")
    print(f"Windowing: window={win_cfg['window_ms']}ms, step={win_cfg['step_ms']}ms, edge_trim={edge_trim} ventanas")
    print(f"Features activos: {feature_flags}")
    print("================================================")

    # ========================================================
    # LOOP SUJETOS
    # ========================================================

    for file_path in tqdm(fif_files):

        subject_id = os.path.basename(
            file_path
        ).replace('_epo.fif', '')

        # ----------------------------------------------------
        # SKIP SI YA EXISTEN LOS ARCHIVOS DEL SUJETO
        # ----------------------------------------------------

        out_features = os.path.join(out_dir, f"{subject_id}_features.npy")
        out_lengths  = os.path.join(out_dir, f"{subject_id}_lengths.npy")

        if os.path.exists(out_features) and os.path.exists(out_lengths):
            continue

        # ----------------------------------------------------
        # CARGAR EPOCHS
        # ----------------------------------------------------

        epochs = mne.read_epochs(
            file_path,
            verbose=False
        )

        # Shape:
        # (n_epochs, n_channels, n_samples)

        data = epochs.get_data()

        # ----------------------------------------------------
        # FEATURES POR ÉPOCA
        # ----------------------------------------------------

        all_epochs_features = []

        lengths = []

        for ep_idx in range(data.shape[0]):

            epoch_data = data[ep_idx]

            ep_feats = extractor.compute_epoch_features(
                epoch_data=epoch_data,
                sfreq=fs,
                win_samples=win_samples,
                step_samples=step_samples,
                edge_trim_windows=edge_trim,
            )

            all_epochs_features.append(ep_feats)

            lengths.append(ep_feats.shape[0])

        # ----------------------------------------------------
        # CONCATENAR SESIÓN
        # ----------------------------------------------------

        X_session = np.vstack(all_epochs_features)

        # ====================================================
        # GUARDADO
        # ====================================================

        np.save(
            os.path.join(
                out_dir,
                f"{subject_id}_features.npy"
            ),
            X_session
        )

        np.save(
            os.path.join(
                out_dir,
                f"{subject_id}_lengths.npy"
            ),
            np.array(lengths)
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n¡Extracción modular finalizada con éxito!")

    print("Listo para PCA y entrenamiento HMM.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()