# scripts/decoding/decode_feature_hmm.py
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import PROJECT_ROOT, clean_path

import numpy as np
import pandas as pd
import joblib
import yaml

from src.decoding.metrics import (
    calculate_fractional_occupancy,
    calculate_dwell_times,
    get_state_stats_per_subject,
)


def _transition_rate(viterbi_path, fs_windows):
    """Numero de cambios de estado por segundo."""
    n_transitions = int(np.sum(np.diff(viterbi_path) != 0))
    total_seconds = len(viterbi_path) / fs_windows
    return n_transitions / total_seconds if total_seconds > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Decoding de estados HMM: FO, Dwell Time y Transition Rate por sujeto."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Ruta al YAML del experimento."
    )
    args = parser.parse_args()

    # ================================================================
    # CARGAR CONFIG
    # ================================================================

    with open(args.config, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    exp_name  = exp_cfg['experiment']['name']
    k_states  = exp_cfg['pipeline']['hmm']['k_states']
    step_ms   = exp_cfg['windowing']['step_ms']

    features_dir = clean_path(exp_cfg['paths']['features_dir'])
    out_dir_base = clean_path(exp_cfg['paths']['output_dir'])
    exp_dir      = os.path.join(out_dir_base, exp_name)

    # fs_windows: cuantas ventanas por segundo.
    # Con step=100ms -> 10 ventanas/s.
    fs_windows = 1000.0 / step_ms

    print(f"\n=== DECODING: {exp_name} ===")
    print(f"K estados    : {k_states}")
    print(f"step_ms      : {step_ms} ms  ->  fs_windows = {fs_windows:.1f} Hz")
    print(f"Features dir : {features_dir}")
    print(f"Exp dir      : {exp_dir}")

    # ================================================================
    # CARGAR MODELO Y VITERBI
    # ================================================================

    model_path   = os.path.join(exp_dir, f'hmm_model_k{k_states}.pkl')
    viterbi_path = os.path.join(exp_dir, f'viterbi_paths_k{k_states}.npy')

    if not os.path.exists(model_path):
        print(f"\nERROR ERROR: Modelo no encontrado:\n   {model_path}")
        print("   Ejecuta primero la etapa 'hmm' o usa --from hmm")
        sys.exit(1)

    if not os.path.exists(viterbi_path):
        print(f"\nERROR ERROR: Viterbi paths no encontrados:\n   {viterbi_path}")
        sys.exit(1)

    model       = joblib.load(model_path)
    viterbi_all = np.load(viterbi_path)

    print(f"\nModelo cargado  : {model_path}")
    print(f"Viterbi shape   : {viterbi_all.shape}  ({viterbi_all.shape[0]} ventanas totales)")

    # ================================================================
    # RECONSTRUIR LONGITUDES POR SUJETO
    #
    # lengths.npy del PCA es por epoca (una entrada por epoca de cada
    # sujeto). Para el decoding necesitamos la longitud TOTAL por sujeto
    # (suma de sus epocas). Los *_lengths.npy individuales de features_dir
    # son la fuente de verdad.
    # ================================================================

    length_files = sorted(glob.glob(os.path.join(features_dir, '*_lengths.npy')))

    if not length_files:
        print(f"\nERROR ERROR: No se encontraron *_lengths.npy en:\n   {features_dir}")
        print("   Ejecuta primero la etapa 'extract'.")
        sys.exit(1)

    subject_ids      = [os.path.basename(f).replace('_lengths.npy', '') for f in length_files]
    subject_lengths  = [int(np.sum(np.load(f))) for f in length_files]

    total_windows = sum(subject_lengths)
    if total_windows != len(viterbi_all):
        print(
            f"\nERROR ERROR: Desincronizacion entre features y Viterbi.\n"
            f"   Suma de longitudes por sujeto : {total_windows}\n"
            f"   Longitud del Viterbi global   : {len(viterbi_all)}\n"
            f"   Causa probable: el Viterbi fue generado con diferentes features.\n"
            f"   Solucion: re-ejecuta desde la etapa 'extract'."
        )
        sys.exit(1)

    print(f"Sujetos encontrados : {len(subject_ids)}")
    print(f"Ventanas totales    : {total_windows} OK")

    # ================================================================
    # CALCULAR METRICAS POR SUJETO
    # ================================================================

    print("\nCalculando metricas por sujeto...")

    rows    = []
    cursor  = 0

    for subj_id, n_windows in zip(subject_ids, subject_lengths):

        path = viterbi_all[cursor:cursor + n_windows]
        cursor += n_windows

        fo    = calculate_fractional_occupancy(path, k_states)
        dwell = calculate_dwell_times(path, fs_windows)
        tr    = _transition_rate(path, fs_windows)

        row = {'subject_id': subj_id, 'n_windows': n_windows}

        for s in range(k_states):
            row[f'FO_S{s}']        = float(fo[s])
            row[f'Dwell_S{s}_sec'] = float(dwell.get(s, 0.0))

        row['transition_rate_per_sec'] = float(tr)
        rows.append(row)

    df_stats = pd.DataFrame(rows)

    # ================================================================
    # GUARDAR RESULTADOS
    # ================================================================

    stats_path      = os.path.join(exp_dir, 'state_stats.csv')
    transmat_path   = os.path.join(exp_dir, f'transition_matrix_k{k_states}.npy')
    summary_path    = os.path.join(exp_dir, 'decoding_summary.json')

    df_stats.to_csv(stats_path, index=False)

    np.save(transmat_path, model.transmat_)

    summary = {
        'experiment'          : exp_name,
        'k_states'            : k_states,
        'n_subjects'          : len(subject_ids),
        'step_ms'             : step_ms,
        'fs_windows_hz'       : fs_windows,
        'total_windows'       : total_windows,
        'global_FO'           : {
            f'S{s}': float(df_stats[f'FO_S{s}'].mean())
            for s in range(k_states)
        },
        'global_dwell_sec'    : {
            f'S{s}': float(df_stats[f'Dwell_S{s}_sec'].mean())
            for s in range(k_states)
        },
        'global_transition_rate_per_sec': float(df_stats['transition_rate_per_sec'].mean()),
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ================================================================
    # RESUMEN EN CONSOLA
    # ================================================================

    print(f"\n{'='*50}")
    print(f"  DECODING COMPLETADO -- {exp_name}")
    print(f"{'='*50}")
    print(f"  Sujetos procesados : {len(subject_ids)}")
    print(f"\n  Ocupacion Fraccional promedio:")
    for s in range(k_states):
        print(f"    Estado {s}: {summary['global_FO'][f'S{s}']:.3f}")
    print(f"\n  Dwell Time promedio (s):")
    for s in range(k_states):
        print(f"    Estado {s}: {summary['global_dwell_sec'][f'S{s}']:.3f} s")
    print(f"\n  Transition rate   : {summary['global_transition_rate_per_sec']:.3f} trans/s")
    print(f"\n  Archivos guardados en: {exp_dir}")
    print(f"    state_stats.csv")
    print(f"    transition_matrix_k{k_states}.npy")
    print(f"    decoding_summary.json")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
