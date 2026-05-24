# src/visualization/signals.py
import numpy as np

def plot_random_epoch(epochs, title="Random Task Epoch"):
    """
    Selecciona y grafica una época aleatoria de un objeto mne.EpochsArray.
    Ideal para inspección visual rápida de ruido o ERPs.
    """
    n_epochs = len(epochs)
    if n_epochs == 0:
        raise ValueError("El objeto Epochs está vacío.")
        
    random_idx = np.random.randint(0, n_epochs)
    print(f"Inspeccionando Epoch #{random_idx}")
    
    # Graficar usando el motor de MNE
    epochs[random_idx].plot(scalings='auto', title=f"{title} (Epoch {random_idx})", show=True)
    return random_idx