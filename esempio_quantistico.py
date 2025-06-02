import numpy as np
from cervelloPensante import CervelloPensante
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Funzione per visualizzare i dati e le predizioni
def plot_results(X, y, predictions=None, title="Data"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', label='Data')
    
    if predictions is not None:
        # Converti le predizioni in classi
        pred_classes = np.argmax(predictions, axis=1)
        plt.scatter(X[:, 0], X[:, 1], c=pred_classes, cmap='viridis', marker='x', alpha=0.5, label='Predictions')
    
    plt.title(title)
    plt.legend()
    plt.show()

# Genera un dataset di esempio con forma di luna (non lineare)
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Visualizza i dati originali
plot_results(X, y, title="Dataset Originale")

# Inizializza il cervello quantistico
cp = CervelloPensante()

# Esegui il training con il circuito quantistico
print("\nAvvio del training quantistico...")
results = cp.usaCervelloQuantistico(
    input_data=X,
    labels=y,
    n_classes=2,  # Poiché abbiamo 2 classi nel dataset
    epochs=200,
    learning_rate=0.05,
    validation_split=0.2,
    regularization=0.01,
    layers=3,  # Usa 3 layer quantistici
    entanglement='full'
)

# Visualizza le prestazioni
print("\nPerformance del modello:")
print(f"Costo finale: {results['final_cost']:.4f}")
print(f"Costo validation: {results['validation_cost']:.4f}")

# Calcola l'accuratezza
predictions = np.array(results['train_predictions'])
pred_classes = np.argmax(predictions, axis=1)
accuracy = accuracy_score(y, pred_classes)
print(f"\nAccuratezza: {accuracy:.2%}")

# Visualizza i risultati
plot_results(X, y, predictions, title="Predizioni del Modello Quantistico")

# Visualizza l'evoluzione del costo durante il training
plt.figure(figsize=(8, 6))
plt.plot(results['train_costs'], label='Training Cost')
if results['val_costs']:
    plt.plot(results['val_costs'], label='Validation Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Evoluzione del Costo durante il Training')
plt.legend()
plt.show()

# Salva i risultati in un file JSON
import json
with open('risultati_quantistici.json', 'w') as f:
    json.dump({
        'quantum_parameters': results['quantum_parameters'],
        'accuracy': float(accuracy),
        'final_cost': float(results['final_cost']),
        'validation_cost': float(results['validation_cost']),
        'configuration': {
            'n_classes': results['n_classes'],
            'layers': results['layers'],
            'entanglement': results['entanglement'],
            'epochs': results['epochs'],
            'learning_rate': results['learning_rate'],
            'regularization': results['regularization']
        }
    }, f, indent=4)

print("\nRisultati salvati in 'risultati_quantistici.json'")
