# src/decoding/metrics.py
import numpy as np
import pandas as pd

def calculate_fractional_occupancy(viterbi_path, n_states):
    """Proporción de tiempo que se pasa en cada estado."""
    counts = np.bincount(viterbi_path, minlength=n_states)
    return counts / len(viterbi_path)

def calculate_dwell_times(viterbi_path, fs_windows):
    """
    Duración promedio de las visitas a cada estado.
    fs_windows: frecuencia de las ventanas (ej. si el step era 20ms, fs=50Hz)
    """
    # Encontrar cambios de estado
    states = np.array(viterbi_path)
    diffs = np.diff(states) != 0
    change_indices = np.where(diffs)[0] + 1
    
    # Dividir el path en "visitas"
    runs = np.split(states, change_indices)
    
    # Agrupar duraciones por estado
    durations = {s: [] for s in np.unique(states)}
    for run in runs:
        durations[run[0]].append(len(run) / fs_windows)
    
    # Promediar
    mean_dwell = {s: np.mean(d) if len(d)>0 else 0 for s, d in durations.items()}
    return mean_dwell

def get_state_stats_per_subject(viterbi_paths, lengths, n_states, fs_windows):
    """Genera un DataFrame con FO y Dwell Time por cada sujeto."""
    start = 0
    subject_data = []
    
    for i, length in enumerate(lengths):
        path = viterbi_paths[start:start+length]
        fo = calculate_fractional_occupancy(path, n_states)
        dwell = calculate_dwell_times(path, fs_windows)
        
        row = {'subject_idx': i}
        for s in range(n_states):
            row[f'FO_S{s}'] = fo[s]
            row[f'Dwell_S{s}'] = dwell.get(s, 0)
        
        subject_data.append(row)
        start += length
        
    return pd.DataFrame(subject_data)