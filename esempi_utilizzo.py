"""
Esempi pratici di utilizzo del DigitalTwin con tutte le sue funzionalità avanzate.
"""

from DigitalTwin import DigitalTwin
from datetime import datetime
import os

# Creazione di un DigitalTwin con un contesto specifico
dt = DigitalTwin("AI Researcher", caricamento=True)  # Tenta di caricare uno stato salvato

# Se non esiste uno stato salvato, procedi con l'inizializzazione normale
if not os.path.exists(f"digital_twins/AI Researcher"):  # Verifica se esiste la directory dello stato
    # Impostazione del contesto
    context = {
        'ambiente': 'laboratorio di ricerca AI',
        'periodo_storico': 'rivoluzione digitale',
        'luogo': 'Silicon Valley',
        'social': {
            'relazioni': ['collega', 'mentore', 'ricercatore'],
            'comunità': 'AI Research Community'
        }
    }
    dt.aggiornaContesto(context)

    # Apprendimento iniziale
    dt.impara("intelligenza artificiale", {
        "relazioni": ["machine learning", "neuroscienze"],
        "bias": {
            "conferma": 0.8,
            "disponibilità": 0.7
        }
    })

    # Salva lo stato dopo l'inizializzazione
    dt.salvaStato()

# 1. Impostazione di un contesto complesso
context = {
    'ambiente': 'laboratorio di ricerca AI',
    'periodo_storico': 'rivoluzione digitale',
    'luogo': 'Silicon Valley',
    'social': {
        'relazioni': ['collega', 'mentore', 'ricercatore'],
        'comunità': 'AI Research Community'
    }
}
dt.aggiornaContesto(context)

# 2. Apprendimento con bias cognitivi
dt.impara("intelligenza artificiale", {
    "relazioni": ["machine learning", "neuroscienze"],
    "bias": {
        "conferma": 0.8,  # Bias di conferma più forte
        "disponibilità": 0.7  # Bias di disponibilità
    }
})

# 3. Generazione di scenari con logica modale
print("\nScenari futuri di sviluppo AI:")
scenari = dt.ipotizzaScenari("sviluppo AI", "futuro")
for scenario in scenari:
    print(f"Scenario: {scenario['scenario']}")
    print(f"Probabilità: {scenario['probabilità']:.2f}")

# 4. Riflessione meta-cognitiva
print("\nRiflessione meta-cognitiva:")
riflessione = dt.rifletteSuSeStesso()
print(f"Conoscenze: {riflessione['consapevolezza']}")
print(f"Domanda esistenziale: {riflessione['domanda_esistenziale']}")
print(f"Stato attuale: {riflessione['stato_attuale']}")
print(f"Bias attivi: {riflessione['bias_attivi']}")

# 5. Combinazione di conoscenze da diversi campi
dt.impara("neuroscienze", {"relazioni": ["biologia", "psicologia"]})
dt.impara("filosofia della mente", {"relazioni": ["filosofia", "psicologia"]})

# 6. Creazione di nuovo sapere attraverso combinazione
print("\nNuovo sapere creato:")
nuovo_sapere = dt.creaNuovoSapere()
for idea in nuovo_sapere:
    print(f"- {idea['nuova_idea']}")

# 7. Analisi del contesto sociale
print("\nAnalisi del contesto sociale:")
print(f"Comunità: {dt.contesto['social']['comunità']}")
print(f"Relazioni: {', '.join(dt.contesto['social']['relazioni'])}")

# 8. Tracciamento delle attività
print("\nCronologia delle attività:")
for evento in dt.storia:
    print(f"{evento['timestamp']}: {evento['azione']}")

# 9. Esempio di salvataggio e caricamento dello stato
print("\nEsempio di salvataggio e caricamento dello stato:")

# Salva lo stato attuale
print("\nSalvataggio dello stato attuale...")
dt.salvaStato()

# Crea un nuovo DigitalTwin e carica lo stato salvato
print("\nCreazione di un nuovo DigitalTwin e caricamento dello stato salvato...")
dt_nuovo = DigitalTwin("AI Researcher", caricamento=True)

print("\nConfronto tra i due DigitalTwin:")
print("Conoscenze originale:", len(dt.conoscenza))
print("Conoscenze nuovo:", len(dt_nuovo.conoscenza))

# Aggiungi una nuova conoscenza e salva di nuovo
print("\nAggiunta di una nuova conoscenza e salvataggio...")
dt.impara("nuova_conoscenza", {"relazioni": ["tecnologia", "innovazione"]})
dt.salvaStato()

# 9. Esempio di pensiero creativo
print("\nEsempio di pensiero creativo:")
idee = dt.pensa("sviluppo futuro dell'IA")
for idea in idee:
    print(f"- {idea}")

# 10. Esempio di ricerca approfondita
print("\nEsempio di ricerca approfondita:")
risultati = dt.ricerca("neuroscienze e IA", approfondimento=True)
for risultato in risultati:
    print(f"Argomento: {risultato['argomento']}")
    print(f"Informazioni: {risultato['informazioni']}")
