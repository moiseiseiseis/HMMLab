# scripts/preprocessing/preprocess_task_batch.py
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT

import yaml
from tqdm import tqdm
from src.data.discovery import discover_task_files
from src.data.loader import load_task_epochs
from src.preprocessing.core import apply_feature_preprocessing


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento por lotes: archivos .txt crudos → épocas .fif limpias."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Ruta al YAML del experimento (requerido por el orquestador; "
             "el preprocesamiento lee sus propias configs globales)."
    )
    args = parser.parse_args()

    # ================================================================
    # CONFIGS GLOBALES
    # El preprocesamiento es idéntico para todos los experimentos
    # Feature-HMM. Sus parámetros viven en configs dedicados,
    # no en el YAML del experimento.
    # ================================================================

    data_cfg_path = os.path.join(PROJECT_ROOT, 'configs/experiments/data_config.yaml')
    pre_cfg_path  = os.path.join(PROJECT_ROOT, 'configs/preprocessing/task_feature_hmm.yaml')

    with open(data_cfg_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    with open(pre_cfg_path, 'r', encoding='utf-8') as f:
        pre_cfg = yaml.safe_load(f)

    # ================================================================
    # RUTAS
    # ================================================================

    raw_dir = data_cfg['dataset']['root_dir']
    out_dir = os.path.join(PROJECT_ROOT, 'data/interim/preprocessed/task/')
    os.makedirs(out_dir, exist_ok=True)

    # ================================================================
    # INFO INICIAL
    # ================================================================

    print("\n=== PREPROCESAMIENTO POR LOTES: raw → .fif ===")
    print(f"Datos crudos : {raw_dir}")
    print(f"Salida       : {out_dir}")

    pre = pre_cfg['preprocessing']
    print(f"Parámetros   : ref={pre['reference']['type']}, "
          f"bandpass=[{pre['filtering']['highpass']}, {pre['filtering']['lowpass']}] Hz, "
          f"resample={pre['resampling']['target']} Hz, "
          f"artefacto<{pre['artifact_removal']['eeg_limit']} V")

    # ================================================================
    # DESCUBRIMIENTO DE ARCHIVOS
    # ================================================================

    df = discover_task_files(raw_dir)

    if df.empty:
        print(f"\n❌ ERROR: No se encontraron archivos en:\n   {raw_dir}")
        print("   Verifica la ruta en configs/experiments/data_config.yaml")
        sys.exit(1)

    print(f"Archivos descubiertos: {len(df)} sujetos")
    print("===============================================\n")

    # ================================================================
    # LOOP DE PREPROCESAMIENTO
    # ================================================================

    errores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            epochs_raw = load_task_epochs(
                row['ruta'],
                data_cfg['dataset']['sampling_rate'],
                data_cfg['channels']['eeg'],
            )

            epochs_clean = apply_feature_preprocessing(epochs_raw, pre_cfg)

            out_name = f"{row['sujeto']}_{row['contexto']}_epo.fif"
            out_path = os.path.join(out_dir, out_name)
            epochs_clean.save(out_path, overwrite=True, verbose=False)

        except Exception as e:
            errores.append((row['sujeto'], str(e)))

    # ================================================================
    # RESUMEN FINAL
    # ================================================================

    n_ok = len(df) - len(errores)
    print(f"\n✅ Preprocesados: {n_ok}/{len(df)}")

    if errores:
        print(f"⚠️  Errores ({len(errores)}):")
        for sujeto, msg in errores:
            print(f"   {sujeto}: {msg}")
        sys.exit(1)

    print(f"Archivos .fif guardados en: {out_dir}")


if __name__ == "__main__":
    main()
