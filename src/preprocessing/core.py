# src/preprocessing/core.py
import mne

def apply_feature_preprocessing(epochs, config):
    """
    Aplica el preprocesamiento fisiológico estricto para Feature-HMM.
    Lee directamente del diccionario anidado bajo la llave 'preprocessing'.
    """
    epochs_clean = epochs.copy()
    
    # Atajo para no escribir config['preprocessing'] todo el tiempo
    pre_cfg = config['preprocessing'] 
    
    # 1. Referencia Promedio (CAR) para centralizar la varianza espacial
    epochs_clean.set_eeg_reference(pre_cfg['reference']['type'], projection=False, verbose=False)
    
    # 2. Filtrado Pasa-Banda (1-30 Hz para potencias y Hjorth)
    epochs_clean.filter(
        l_freq=pre_cfg['filtering']['highpass'], 
        h_freq=pre_cfg['filtering']['lowpass'], 
        fir_design='firwin',
        phase=pre_cfg['filtering']['phase'],
        verbose=False
    )
    
    # 3. Downsampling (Ahorro masivo de RAM y CPU para las Features)
    if pre_cfg['resampling']['apply']:
        target_sfreq = pre_cfg['resampling']['target']
        epochs_clean.resample(target_sfreq, verbose=False)
    
    # 4. Limpieza de Artefactos Extremos (El escudo contra el sujeto EDGSSINGO)
    reject_criteria = dict(eeg=float(pre_cfg['artifact_removal']['eeg_limit']))
    epochs_clean.drop_bad(reject=reject_criteria, verbose=False)
    
    return epochs_clean