# Software Digital Twin
# di Luigi Usai
# 2 giugno 2025
# Quartucciu (CA), Italy

Software Digital Twin è un sistema avanzato di intelligenza artificiale che implementa un Digital Twin con caratteristiche cognitive avanzate. Il sistema è stato progettato per simulare il funzionamento della mente umana attraverso l'implementazione di bias cognitivi e un sistema di memoria sofisticato.

## Caratteristiche principali

- **Bias cognitivi integrati**: Il sistema implementa bias cognitivi realistici come:
  - Bias di conferma (confirmation bias)
  - Bias di ancoraggio (anchoring bias)
  - Bias di disponibilità (availability bias)
  - Bias di avversione alla perdita (loss aversion)

- **Sistema di memoria avanzato**:
  - Memoria di lavoro (Working Memory) per concetti attivi
  - Memoria episodica per eventi e interazioni
  - Memoria semantica per la conoscenza strutturata

- **Capacità cognitive avanzate**:
  - Apprendimento automatico di nuovi concetti
  - Generazione di relazioni semantiche
  - Pensiero contestuale
  - Modulazione di stato interno basata su bias

- **Gestione del contesto**:
  - Contesto situato
  - Integrazione con ambiente e periodo storico
  - Gestione di relazioni sociali
  - Tracciamento temporale

## Requisiti

- Python 3.13 o superiore
- Accesso a un LLM (Language Model) compatibile
- Pacchetti Python elencati in requirements.txt

## Installazione

1. Clonare il repository:
```bash
git clone [url_del_repository]
cd Software Digital Twin
```

2. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo di base

```python
from digital_twin import DigitalTwin

dt = DigitalTwin("MioDigitalTwin")

# Apprendimento
dt.impara("nuovo_concetto")

# Pensiero
dt.pensa("argomento")

# Memoria
dt.memorizza({"chiave": "valore"})
```

## Struttura del progetto

```
Software Digital Twin/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── digital_twin/
│   ├── __init__.py
│   ├── DigitalTwin.py
│   ├── KnowledgeStore.py
│   ├── LLMManager.py
│   ├── MemoryManager.py
│   └── tests/
└── docs/
```

## Licenza

Questo software è rilasciato sotto licenza MIT. Consultare il file LICENSE per maggiori dettagli.

## Contribuire

Contributi sono benvenuti! Per contribuire:
1. Fork il repository
2. Crea una branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push alla branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Supporto

Per supporto o domande, contattare: luigi.usai@email.com

## Descrizione tecnica dettagliata

Il Software Digital Twin implementa un sistema di intelligenza artificiale che simula il funzionamento della mente umana attraverso:

1. **Bias cognitivi**:
   - Il sistema modella bias cognitivi realistici che influenzano l'apprendimento e il ragionamento
   - Ogni bias ha un peso numerico che modula l'importanza dei nuovi concetti
   - I bias interagiscono tra loro per creare un comportamento più naturale

2. **Sistema di memoria**:
   - Memoria di lavoro limitata (7 elementi) per simulare la memoria operativa umana
   - Memoria episodica per tracciare eventi e interazioni
   - Memoria semantica per la conoscenza strutturata
   - Sistema di priorità per la gestione della memoria di lavoro

3. **Gestione del contesto**:
   - Contesto situato che include ambiente, periodo storico e luogo
   - Gestione di relazioni sociali e interazioni
   - Tracciamento temporale per la coerenza delle risposte
   - Contesto linguistico e culturale

4. **Apprendimento e ragionamento**:
   - Apprendimento automatico di nuovi concetti
   - Generazione di relazioni semantiche tra concetti
   - Pensiero contestuale basato sullo stato interno
   - Modulazione di stato basata su bias e contesto

## Possibili applicazioni

- Simulazione di comportamenti umani in ambienti virtuali
- Sistema di supporto alla decisione con bias cognitivi
- Studio dei processi decisionali umani
- Simulazione di interazioni sociali
- Sviluppo di assistenti virtuali più naturali

## Note tecniche

Il sistema è stato progettato per essere estensibile e modularizzato, permettendo l'aggiunta di nuovi bias cognitivi, tipi di memoria o meccanismi di ragionamento. La struttura del codice è organizzata per facilitare la manutenzione e l'evoluzione del progetto.
