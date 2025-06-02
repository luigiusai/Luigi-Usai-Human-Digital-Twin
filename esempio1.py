from DigitalTwin import DigitalTwin

# Creazione del DigitalTwin
dt = DigitalTwin("AI Researcher")

# Generazione di scenari
scenari = dt.genera_scenari("intelligenza artificiale", livello_creatività=0.8)
for scenario in scenari:
    print(f"\nScenario: {scenario['narrativa']}")
    print(f"Probabilità: {scenario['probabilità']:.2f}")
    print("Attori principali:")
    for attore in scenario['attori']:
        print(f"- {attore['ruolo']}: {attore['descrizione']}")
    print("Implicazioni etiche:")
    for implicazione in scenario['implicazioni_etiche']:
        print(f"- {implicazione}")

# Transizione tra livelli
domanda_critica = "Come l'IA può influenzare la società?"
if dt.transizioneLivelli("ontologico", "epistemico", domanda_critica):
    print("Transizione avvenuta con successo")
else:
    print("Transizione non riuscita")

# Modulazione dei bias
dt.stato['energia'] = 80
dt.stato['creatività'] = 90
dt.modulaBias()
print(f"Bias aggiornati: {dt.bias}")