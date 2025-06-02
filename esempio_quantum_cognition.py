import numpy as np
from cervelloPensante import CervelloPensante
import matplotlib.pyplot as plt

def plot_quantum_metrics(metrics):
    """Visualizza le metriche quantistiche."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot delle emozioni
    emotions = metrics['emotions']
    ax1.bar(emotions.keys(), emotions.values())
    ax1.set_title('Metriche Emotive Quantistiche')
    ax1.set_ylabel('Valore')
    
    # Plot delle metriche di memoria
    memory = metrics['memory_metrics']
    ax2.bar(memory.keys(), memory.values())
    ax2.set_title('Metriche di Memoria Quantistica')
    ax2.set_ylabel('Valore')
    
    plt.tight_layout()
    plt.show()

# Inizializza il cervello quantistico
cp = CervelloPensante()

# Esempio di stimolo
stimolo = "Devo prendere una decisione importante"

# Processa lo stimolo con cognizione quantistica
risultato = cp.pensa(stimolo)

# Estrai le metriche quantistiche
quantum_metrics = risultato['quantum_metrics']

# Visualizza le metriche
print("\nMetriche della Decisione Quantistica:")
print(f"Valore di Decisione: {quantum_metrics['decision']:.4f}")

print("\nStato Emotivo:")
for emo, val in quantum_metrics['emotions'].items():
    print(f"{emo}: {val:.4f}")

print("\nMetriche di Memoria:")
for mem, val in quantum_metrics['memory_metrics'].items():
    print(f"{mem}: {val:.4f}")

# Visualizza graficamente
plot_quantum_metrics(quantum_metrics)
