import json
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from core.base_manager import BaseManager
from llm_integration.llm_manager import LLMManager, LLMProvider, LLMConfig

from helpers.knowledge_manager import KnowledgeManager
from helpers.memory_system import MemorySystem
from helpers.cognitive_biases_manager import CognitiveBiasesManager
from helpers.scenario_generator import ScenarioGenerator

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

class DigitalTwin(BaseManager):
    """Classe principale del Digital Twin."""
    
    nome: str = "DigitalTwin"
    contesto: Dict[str, Any] = {
        'ambiente': 'virtuale',
        'periodo_storico': 'postumano',
        'lingua': 'italiano',
        'luogo': None,
        'tempo': Field(default_factory=datetime.now),
        'social': {'relazioni': [], 'comunità': None}
    }
    
    def __init__(self, nome: str = "DigitalTwin", llm_config: Optional[LLMConfig] = None):
        """Inizializza il Digital Twin."""
        # Inizializza LLMManager
        self.llm_manager = LLMManager(
            llm_config or LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                temperature=0.7
            )
        )
        
        # Inizializza i gestori centralizzati
        self.knowledge_manager = KnowledgeManager(self.llm_manager)
        self.memory_system = MemorySystem(self.llm_manager)
        self.biases_manager = CognitiveBiasesManager(self.llm_manager)
        self.scenario_generator = ScenarioGenerator(self.llm_manager)
        
        # Inizializza il cervello pensante
        self.cervello_pensante = CervelloPensante("Cervello", self)
        
        # Imposta il nome e il contesto
        self.nome = nome
        self.contesto = {
            'ambiente': 'virtuale',
            'periodo_storico': 'postumano',
            'lingua': 'italiano',
            'luogo': None,
            'tempo': datetime.now(),
            'social': {'relazioni': [], 'comunità': None}
        }
        
        # Aggiungi l'inizializzazione alla cronologia
        self.add_to_history({
            'tipo': 'inizializzazione',
            'nome': nome,
            'config': llm_config.dict() if llm_config else None,
            'contesto': self.contesto
        })
        
    def pensa(self, stimolo: str) -> Dict[str, Any]:
        """Processa uno stimolo attraverso il cervello pensante."""
        # Genera l'embedding semantico dello stimolo
        stimolo_embedding = self.llm_manager.generate_embedding(stimolo)
        
        # Processa con il cervello pensante
        risposta = self.cervello_pensante.pensa(stimolo)
        
        # Genera una risposta linguistica con l'LLM
        context = {
            'stimolo': stimolo,
            'risposta_quantistica': risposta,
            'contesto': self.contesto
        }
        
        # Genera una risposta linguistica
        risposta_linguistica = self.llm_manager.generate(
            f"""
            Basato sulla risposta quantistica e sul contesto, genera una risposta linguistica coerente.
            Risposta Quantistica: {risposta}
            Contesto: {self.contesto}
            """
        )
        
        # Aggiungi alla cronologia
        self.add_to_history({
            'tipo': 'pensiero',
            'stimolo': stimolo,
            'embedding': stimolo_embedding,
            'risposta_quantistica': risposta,
            'risposta_linguistica': risposta_linguistica
        })
        
        return {
            'risposta_quantistica': risposta,
            'risposta_linguistica': risposta_linguistica
        }
        
        # Aggiungi il contesto alla cronologia
        self.add_to_history({
            'tipo': 'inizializzazione',
            'nome': nome,
            'config': llm_config.dict() if llm_config else None,
            'contesto': self.contesto
        })

    def impara(self, nuovo_concetto: str, relazioni: Optional[List[str]] = None) -> bool:
        """Impara un nuovo concetto."""
        try:
            # Verifica se il concetto esiste già
            if self.knowledge_manager.retrieve(nuovo_concetto):
                print(f"Attenzione: il concetto '{nuovo_conceto}' è già stato memorizzato.")
                return False
                
            # Se non sono fornite relazioni, generale automaticamente
            if relazioni is None:
                relazioni = self.knowledge_manager.generate_relations(nuovo_concetto)
                
            # Memorizza il concetto
            self.knowledge_manager.store(nuovo_concetto, {
                'relazioni': relazioni,
                'importanza': 1.0,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'fonte': 'apprendimento'
                }
            })
            
            # Aggiorna la memoria
            self.memory_system.add_to_working_memory({
                'azione': 'impara',
                'concetto': nuovo_concetto
            })
            
            # Aggiorna lo stato interno
            self.biases_manager.update_state('learn', 1.0)
            
            return True
            
        except Exception as e:
            print(f"Errore durante l'apprendimento: {str(e)}")
            return False

    def pensa(self, argomento: str) -> Dict[str, Any]:
        """Pensa su un argomento."""
        try:
            # Aggiungi l'argomento alla memoria di lavoro
            self.memory_system.add_to_working_memory({
                'azione': 'pensa',
                'argomento': argomento
            })
            
            # Cerca concetti rilevanti
            relevant_concepts = self.knowledge_manager.search(argomento)
            
            # Genera una risposta usando LLM
            prompt = f"""
            Pensiero su: {argomento}
            Contesto: {json.dumps(self.contesto, ensure_ascii=False)}
            Concetti rilevanti: {[c.concept for c in relevant_concepts]}
            """
            
            risposta = self.llm_manager.generate_response(prompt)
            
            # Registra l'interazione
            self.memory_system.record_event({
                'tipo': 'interaction',
                'argomento': argomento,
                'risposta': risposta
            })
            
            # Aggiorna lo stato interno
            self.biases_manager.update_state('think', 1.0)
            
            return {
                'risposta': risposta,
                'timestamp': datetime.now().isoformat(),
                'stato': self.biases_manager.get_current_state()
            }
            
        except Exception as e:
            print(f"Errore durante il pensiero: {str(e)}")
            return {
                'risposta': "Mi dispiace, c'è stato un errore durante il pensiero.",
                'timestamp': datetime.now().isoformat(),
                'stato': self.biases_manager.get_current_state()
            }

    def memorizza(self, informazione: Dict[str, Any]) -> None:
        """Memorizza una nuova informazione."""
        try:
            # Aggiungi all'episodica
            self.memory_system.record_event(informazione)
            
            # Se è una nuova conoscenza, aggiungi anche alla semantica
            if 'concetto' in informazione:
                self.knowledge_manager.store(informazione['concetto'], informazione)
                
            # Aggiorna lo stato
            self.biases_manager.update_state('memorize', 1.0)
            
        except Exception as e:
            print(f"Errore durante la memorizzazione: {str(e)}")

    def genera_scenari(self, topic: str) -> List[Dict[str, Any]]:
        """Genera scenari su un topic."""
        try:
            # Genera uno scenario base
            base_scenario = self.scenario_generator.generate_scenario(
                topic,
                self.contesto
            )
            
            # Genera variazioni
            variations = self.scenario_generator.generate_variations(base_scenario)
            
            # Valuta la rilevanza
            scenarios = [base_scenario] + variations
            for scenario in scenarios:
                scenario.relevance = self.scenario_generator.evaluate_relevance(
                    scenario,
                    self.contesto
                )
            
            return [s.dict() for s in scenarios]
            
        except Exception as e:
            print(f"Errore durante la generazione di scenari: {str(e)}")
            return []

    def get_stato(self) -> Dict[str, Any]:
        """Restituisce lo stato attuale."""
        return {
            'cognitivo': self.biases_manager.get_current_state(),
            'memoria': {
                'working': self.memory_system.get_working_memory(),
                'episodic': self.memory_system.get_episodic_memory(),
                'semantic': self.memory_system.get_semantic_memory()
            },
            'conoscenza': self.knowledge_manager.get_all_concepts(),
            'contesto': self.contesto
        }

    def save_state(self, directory: str = "stato") -> None:
        """Salva lo stato corrente."""
        try:
            # Crea la directory se non esiste
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Salva lo stato
            with open(f"{directory}/{self.nome}_stato.json", 'w', encoding='utf-8') as f:
                json.dump(self.get_stato(), f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Errore durante il salvataggio dello stato: {str(e)}")

    def load_state(self, directory: str = "stato") -> bool:
        """Carica uno stato precedente."""
        try:
            # Carica lo stato
            with open(f"{directory}/{self.nome}_stato.json", 'r', encoding='utf-8') as f:
                stato = json.load(f)
                
            # Ripristina il contesto
            self.contesto = stato['contesto']
            
            # Ripristina la conoscenza
            for concept in stato['conoscenza']:
                self.knowledge_manager.store(concept, {
                    'metadata': {'timestamp': datetime.now().isoformat()}
                })
                
            # Ripristina la memoria
            for entry in stato['memoria']['working']:
                self.memory_system.add_to_working_memory(entry)
            for entry in stato['memoria']['episodic']:
                self.memory_system.record_event(entry)
            for concept, entries in stato['memoria']['semantic'].items():
                for entry in entries:
                    self.memory_system.store_knowledge(concept, entry)
                    
            # Ripristina lo stato cognitivo
            self.biases_manager.current_state = stato['cognitivo']
            
            return True
            
        except Exception as e:
            print(f"Errore durante il caricamento dello stato: {str(e)}")
            return False

    def reset(self) -> None:
        """Resetta il Digital Twin."""
        self.memory_system.clear_working_memory()
        self.biases_manager.current_state = {
            'energy': 100,
            'creativity': 100,
            'motivation': 100,
            'mood': 'neutral',
            'stress': 0
        }
        self.contesto = {
            'ambiente': 'virtuale',
            'periodo_storico': 'postumano',
            'lingua': 'italiano',
            'luogo': None,
            'tempo': datetime.now(),
            'social': {'relazioni': [], 'comunità': None}
        }

    def __init__(self, nome="DigitalTwin", caricamento=False):
        """Inizializza il gemello digitale."""
        try:
            self.nome = nome
            self.llm_manager = LLMManager()  # Gestore LLM
            self.knowledge_store = KnowledgeStore(self.llm_manager)  # Archivio delle conoscenze
            self.memory_manager = MemoryManager(self.llm_manager)  # Gestore della memoria
            print(f"Knowledge store initialized: {self.knowledge_store}")  # Debug print
            
            # Initialize other attributes
            self.conoscenza = {}  # Database delle conoscenze
            self.stato = {
                'umore': 'neutro',
                'energia': 100,
                'motivazione': 80,
                'creatività': 70
            }
            self.abilita = []      # Lista delle abilità acquisite
            self.scenari = []      # Lista dei scenari ipotetici
            
            # Verifica che i componenti siano stati inizializzati correttamente
            if not all([self.llm_manager, self.knowledge_store, self.memory_manager]):
                raise ValueError("Uno o più componenti non sono stati inizializzati correttamente")
            
        except Exception as e:
            print(f"Errore durante l'inizializzazione: {str(e)}")
            raise
        self.direttorio_persistenza = f"digital_twins/{self.nome}"
        if caricamento:
            self.caricaStato()
        
        # Sistema di contesto situato
        self.contesto = {
            'ambiente': 'virtuale',
            'periodo_storico': 'postumano',
            'lingua': 'italiano',
            'luogo': None,
            'tempo': datetime.now(),
            'social': {
                'relazioni': [],
                'comunità': None
            },
            'conoscenze': []
        }
        
        # Sistema di bias cognitivi
        self.bias = {
            'conferma': 0.7,    # Bias di conferma
            'ancoraggio': 0.5,  # Effetto di ancoraggio
            'disponibilità': 0.6,  # Bias di disponibilità
            'loss_aversion': 0.8,  # Bias di avversione alla perdita
            'modulazione': {  # Sistema di modulazione
                'energia': 0.5,
                'creatività': 0.4,
                'motivazione': 0.3
            }
        }
        
        # Mantenere la compatibilità con i test
        self.memoria = {}
        self.storia = []
        
        # Inizializza la memoria con i valori di base
        self.memoria.update({
            'stato_iniziale': self.stato.copy(),
            'bias': self.bias.copy(),
            'ultima_interazione': None
        })

    def _applica_bias_conferma(self, nuovo_concetto: str) -> float:
        """Calcola il bias di conferma per un nuovo concetto."""
        # Trova concetti simili nella conoscenza esistente
        concetti_simili = self.knowledge_store.search(nuovo_concetto)
        
        # Se ci sono concetti simili, aumenta la fiducia
        if concetti_simili:
            return self.bias['conferma'] * len(concetti_simili)
        return 0.0
    
    def _applica_bias_disponibilità(self, concetto: str) -> float:
        """Calcola il bias di disponibilità per un concetto."""
        # Controlla quando è stata l'ultima volta che il concetto è stato utilizzato
        if concetto in self.memoria:
            ultima_utilizzazione = self.memoria[concetto].get('timestamp', datetime.min)
            tempo_passato = (datetime.now() - ultima_utilizzazione).total_seconds()
            # Concetti più recenti hanno un bias più alto
            return self.bias['disponibilità'] * (1 / (1 + tempo_passato / 86400))  # Normalizzato su 24 ore
        return 0.0
    
    def _modula_stato(self, trigger: str) -> None:
        """Modula lo stato interno in base a trigger specifici."""
        modulazioni = self.bias['modulazione']
        
        if trigger == 'nuova_conoscenza':
            self.stato['energia'] -= modulazioni['energia'] * 10
            self.stato['creatività'] += modulazioni['creatività'] * 5
            self.stato['motivazione'] += modulazioni['motivazione'] * 3
        
        elif trigger == 'contraddizione':
            self.stato['energia'] -= modulazioni['energia'] * 20
            self.stato['creatività'] += modulazioni['creatività'] * 10
            self.stato['motivazione'] -= modulazioni['motivazione'] * 5
        
        # Assicurati che i valori rimangano nel range valido
        for key in self.stato:
            self.stato[key] = max(0, min(100, self.stato[key]))
    
    def _applica_bias_ancoraggio(self, nuovo_concetto: str) -> float:
        """Calcola il bias di ancoraggio per un nuovo concetto."""
        # Se il concetto è simile a uno già conosciuto, mantenere il valore precedente
        concetti_simili = self.knowledge_store.search(nuovo_concetto)
        if concetti_simili:
            return self.bias['ancoraggio'] * sum(
                self.knowledge_store.get_concept_value(c) for c in concetti_simili
            ) / len(concetti_simili)
        return 0.0
    
    def _applica_bias_loss_aversion(self, nuovo_concetto: str) -> float:
        """Calcola il bias di avversione alla perdita."""
        # Se il concetto contraddice qualcosa di conosciuto, aumenta l'avversione
        contraddizioni = self.knowledge_store.find_contraddizioni(nuovo_concetto)
        if contraddizioni:
            return self.bias['loss_aversion'] * len(contraddizioni)
        return 0.0
    
    def impara(self, nuovo_concetto, relazioni=None):
        """Impara un nuovo concetto considerando i bias cognitivi."""
        # Applica tutti i bias
        bias_conferma = self._applica_bias_conferma(nuovo_concetto)
        bias_disponibilità = self._applica_bias_disponibilità(nuovo_concetto)
        bias_ancoraggio = self._applica_bias_ancoraggio(nuovo_concetto)
        bias_loss_aversion = self._applica_bias_loss_aversion(nuovo_concetto)
        
        # Calcola il peso totale del concetto
        peso_concetto = 1.0 + bias_conferma + bias_disponibilità - bias_loss_aversion
        
        # Se il peso è troppo basso, potrebbe non valere la pena imparare
        if peso_concetto < 0.5:
            return False
        
        # Procedi con l'apprendimento
        risultato = self.ricorda(nuovo_concetto)
        if risultato['trovato']:
            return False
        
        if relazioni is None:
            relazioni = self.genera_relazioni_automatiche(nuovo_concetto)
            
        # Salva il concetto con il peso calcolato
        self.knowledge_store.store(nuovo_concetto, {
            'valore': peso_concetto,
            'relazioni': relazioni,
            'bias': {
                'conferma': bias_conferma,
                'disponibilità': bias_disponibilità,
                'ancoraggio': bias_ancoraggio,
                'loss_aversion': bias_loss_aversion
            }
        })
        
        # Aggiorna il contesto
        self.contesto['conoscenze'].append(nuovo_concetto)
        
        # Modula lo stato
        self._modula_stato('nuova_conoscenza')
        
        # Aggiorna la memoria
        self.memory_manager.add_to_working_memory(nuovo_concetto)
        self.memory_manager.add_to_episodic_memory(nuovo_concetto, peso_concetto)
        
        return True
        
        # Sistema di contraddizioni
        self.contraddizioni = {
            'attuali': [],
            'risolte': [],
            'potenziali': []
        }
        
        # Sistema di transizione tra livelli
        self.transizioni = {
            'ontologico_epistemico': [],
            'epistemico_metacognitivo': [],
            'metacognitivo_ontologico': []
        }
        
        # Sistema di livelli di astrazione
        self.livelli_astrazione = {
            'ontologico': [],
            'epistemico': [],
            'metacognitivo': []
        }

    def pensa(self, argomento: str) -> dict:
        """Processa un argomento e genera una risposta."""
        # Aggiorna il contesto con l'argomento
        self.contesto['argomento'] = argomento
        
        # Aggiungi l'argomento alla memoria di lavoro
        self.memory_manager.add_to_working_memory(argomento)
        
        # Genera una risposta
        risposta = self.llm_manager.generate_response(argomento)
        
        # Registra l'interazione nella memoria episodica
        self.memory_manager.record_interaction({
            'type': 'interaction',
            'topic': argomento,
            'response': risposta
        })
        
        # Aggiorna lo stato
        self.stato['creatività'] += 5
        self.stato['energia'] -= 2
        
        return {
            'risposta': risposta,
            'timestamp': datetime.now().isoformat(),
            'stato': self.stato.copy()
        }

    def apprendi(self, argomento, metodo='attivo'):
        """Apprende un argomento specifico attraverso vari metodi."""
        if metodo == 'attivo':
            self.studia(argomento)
        elif metodo == 'passivo':
            self.memorizza(argomento)
        else:
            self.impara(argomento)
        self.stato['energia'] -= 10
        self.stato['motivazione'] += 5

    def chiediInfo(self, domanda):
        """Ricerca informazioni nel proprio database di conoscenza."""
        risultati = []
        for argomento, dati in self.conoscenza.items():
            if domanda.lower() in argomento.lower():
                risultati.append({
                    'argomento': argomento,
                    'informazioni': dati['data']
                })
        self.stato['energia'] -= 2
        return risultati

    def studia(self, argomento):
        """Analizza approfonditamente un argomento."""
        if argomento in self.conoscenza:
            self.conoscenza[argomento]['livello_conoscenza'] = 'approfondito'
            self.memorizza(argomento)
            self.stato['energia'] -= 15
            self.stato['motivazione'] += 15

    def cerca(self, keyword):
        """Cerca informazioni correlate alla keyword."""
        return self.chiediInfo(keyword)

    def ricerca(self, domanda, approfondimento=True):
        """Esegue una ricerca dettagliata su una domanda."""
        risultati = self.chiediInfo(domanda)
        if approfondimento:
            for r in risultati:
                self.studia(r['argomento'])
        self.stato['energia'] -= 20
        return risultati

    def chiediConsiglio(self, situazione):
        """Fornisce consigli basati sulle conoscenze acquisite."""
        consigli = []
        for argomento in self.conoscenza:
            if situazione.lower() in argomento.lower():
                consigli.append(f"Per {situazione}, considera {self.conoscenza[argomento]}")
        self.stato['energia'] -= 5
        return consigli

    def immagina(self, scenario):
        """Crea scenari immaginari basati su conoscenze esistenti."""
        scenario_ipotetico = self.ipotizzaScenari(scenario)
        self.scenari.append(scenario_ipotetico)
        self.stato['creatività'] += 10
        return scenario_ipotetico

    def sogna(self):
        """Genera pensieri e idee creative."""
        return self.immaginaCosaPuoAccadere()

    def immaginaCosaPuoAccadere(self):
        """Prevede possibili sviluppi futuri."""
        possibili_eventi = []
        for argomento in self.conoscenza:
            possibili_eventi.extend(self.ipotizza(argomento))
        self.stato['creatività'] += 5
        return possibili_eventi

    def memorizza(self, informazione: dict) -> None:
        """Memorizza una nuova informazione."""
        # Aggiorniamo sia la memoria vecchia che la nuova
        self.memoria.update(informazione)
        
        # Aggiungi l'informazione alla memoria di lavoro
        for key, value in informazione.items():
            self.memory_manager.add_to_working_memory(f"{key}:{value}")
        
        # Registra l'evento nella memoria episodica
        self.memory_manager.record_event({
            'type': 'memory',
            'info': informazione
        })
        
        self.stato['creatività'] += 5
        self.stato['energia'] -= 2
        return True

    def crea(self, tipo='idea'):
        """Crea nuove idee o concetti."""
        if tipo == 'idea':
            return self.immagina('nuova idea')
        elif tipo == 'teoria':
            return self.teorizza()
        else:
            return self.combinaSettoriDiversiDelloScibile()

    def ipotizza(self, situazione):
        """Genera ipotesi su una situazione."""
        ipotesi = []
        for argomento in self.conoscenza:
            if situazione.lower() in argomento.lower():
                ipotesi.append(f"{situazione} potrebbe implicare {self.conoscenza[argomento]}")
        self.stato['creatività'] += 5
        return ipotesi

    def deduci(self, dati):
        """Applica il ragionamento deduttivo."""
        conclusioni = []
        for argomento in self.conoscenza:
            if all(d in argomento.lower() for d in dati):
                conclusioni.append(self.conoscenza[argomento])
        self.stato['energia'] -= 10
        return conclusioni

    def inferisci(self, indizi):
        """Applica il ragionamento induttivo."""
        return self.deduci(indizi)

    def teorizza(self):
        """Sviluppa teorie nuove."""
        teoria = self.crea('teoria')
        self.impara(teoria)
        self.stato['creatività'] += 20
        return teoria

    def combinaSettoriDiversiDelloScibile(self, argomento=None):
        """Combina conoscenze da diversi campi."""
        combinazioni = []
        if argomento:
            for altro_argomento in self.conoscenza:
                if argomento != altro_argomento:
                    combinazioni.append({
                        'conoscenza_1': argomento,
                        'conoscenza_2': altro_argomento,
                        'nuova_idea': f"{argomento} + {altro_argomento}"
                    })
        self.stato['creatività'] += 15
        return combinazioni

    def creaNuovoSapere(self):
        """Sviluppa nuovo sapere combinando conoscenze esistenti."""
        nuovo_sapere = self.combinaSettoriDiversiDelloScibile()
        for idea in nuovo_sapere:
            self.impara(idea['nuova_idea'])
        self.stato['creatività'] += 25
        return nuovo_sapere

    def transizioneLivelli(self, livello_partenza: str, livello_arrivo: str, domanda_critica: str) -> bool:
        """
        Gestisce la transizione tra livelli di astrazione.
        
        Args:
            livello_partenza: Il livello di astrazione di partenza
            livello_arrivo: Il livello di astrazione di arrivo
            domanda_critica: La domanda critica per la transizione
            
        Returns:
            True se la transizione è avvenuta con successo, False altrimenti
        """
        # Implementazione della transizione tra livelli
        if livello_partenza in self.livelli_astrazione and livello_arrivo in self.livelli_astrazione:
            # Eseguire la transizione
            self.livelli_astrazione[livello_arrivo].append(domanda_critica)
            return True
        return False

    def genera_scenari(self, concetto: str, livello_creatività: float = 0.7) -> List[Dict]:
        """
        Genera scenari futuri con attributi narrativi, implicazioni etiche e ragionamento modale.
        
        Args:
            concetto: Il concetto base per i scenari
            livello_creatività: Valore tra 0 e 1 che influenza la creatività
            
        Returns:
            Lista di scenari con attributi dettagliati
        """
        scenari = []
        
        # Analisi del contesto
        contesto = {
            'tempo': self.contesto['tempo'],
            'luogo': self.contesto['luogo'],
            'periodo_storico': self.contesto['periodo_storico']
        }
        
        # Genera 3 scenari
        for _ in range(3):
            # Genera ipotesi formali
            ipotesi = {
                'credenza': self.genera_ipotesi(concetto),
                'conoscenza': self.genera_conoscenza(concetto),
                'possibilità': self.genera_possibilità(concetto)
            }
            
            # Crea il dizionario base del scenario
            scenario = {
                'concetto': concetto,
                'attori': self.genera_attori(),
                'contesto': contesto,
                'implicazioni_etiche': self.genera_implicazioni_etiche(),
                'probabilità': self.calcola_probabilità(),
                'tempo': self.genera_tempo(),
                'narrativa': self.genera_narrativa(concetto),
                'ipotesi': ipotesi,
                'timestamp_generazione': datetime.now()
            }
            
            # Aggiungi il controfattuale dopo che scenario è stato creato
            scenario['controfattuale'] = self.valuta_controfattuale(scenario)
            scenari.append(scenario)
        
        # Aggiorna lo stato
        self.stato['creatività'] += 10 * livello_creatività
        self.stato['energia'] -= 15 * livello_creatività
        
        return scenari

    def genera_ipotesi(self, concetto: str) -> str:
        """Genera un'ipotesi formale per il concetto."""
        return f"Per ogni x, se x è {concetto}, allora {self.genera_condizione(concetto)}"

    def genera_conoscenza(self, concetto: str) -> str:
        """Genera una conoscenza formale per il concetto."""
        return f"Sappiamo che {concetto} implica {self.genera_implicazione(concetto)}"

    def genera_possibilità(self, concetto: str) -> str:
        """Genera una possibilità modale per il concetto."""
        return f"È possibile che {concetto} porti a {self.genera_esito(concetto)}"

    def genera_condizione(self, concetto: str) -> str:
        """Genera una condizione formale per il concetto."""
        parole = concetto.lower().split()
        if len(parole) > 1:
            return f"{parole[0]} è correlato a {parole[1]}"
        return f"{concetto} ha un effetto significativo"

    def genera_implicazione(self, concetto: str) -> str:
        """Genera un'implicazione formale per il concetto."""
        parole = concetto.lower().split()
        if len(parole) > 1:
            return f"{parole[1]} influenza {parole[0]}"
        return f"{concetto} implica cambiamenti significativi"

    def genera_esito(self, concetto: str) -> str:
        """Genera un esito formale per il concetto."""
        parole = concetto.lower().split()
        if len(parole) > 1:
            return f"un miglioramento in {parole[1]}"
        return f"un miglioramento in {concetto}"

    def genera_esito_alternativo(self) -> str:
        """Genera un esito alternativo per il ragionamento controfattuale."""
        esiti = [
            "non ci sarebbero stati miglioramenti significativi",
            "il sistema sarebbe rimasto statico",
            "non si sarebbero verificati cambiamenti rilevanti",
            "la situazione sarebbe rimasta invariata"
        ]
        return random.choice(esiti)

    def valuta_controfattuale(self, scenario: Dict) -> str:
        """Valuta un scenario controfattuale."""
        if 'ipotesi' in scenario and 'credenza' in scenario['ipotesi']:
            return f"Se non fosse stato {scenario['ipotesi']['credenza']}, allora {self.genera_esito_alternativo()}"
        return f"Se non fosse stato {scenario.get('concetto', 'questo evento')}, allora {self.genera_esito_alternativo()}"

    def genera_attori(self) -> List[Dict]:
        """Genera attori per lo scenario."""
        attori = []
        ruoli = ['principale', 'secondario', 'antagonista']
        
        for ruolo in ruoli:
            attore = {
                'ruolo': ruolo,
                'descrizione': self.genera_descrizione_attore(ruolo),
                'motivazioni': self.genera_motivazioni(),
                'azioni': self.generaazioni()
            }
            attori.append(attore)
        
        return attori

    def genera_implicazioni_etiche(self) -> List[str]:
        """Genera implicazioni etiche per lo scenario."""
        implicazioni = []
        categorie = ['privacy', 'giustizia', 'benessere', 'diritti']
        
        for categoria in categorie:
            implicazione = f"{categoria}: {self.genera_descrizione_implicazione(categoria)}"
            implicazioni.append(implicazione)
        
        return implicazioni

    def calcola_probabilità(self) -> float:
        """Calcola la probabilità dello scenario."""
        # Usa una distribuzione beta per simulare l'incertezza
        alpha = 2
        beta = 5
        return random.betavariate(alpha, beta)

    def genera_condizioni_temporali(self) -> List[str]:
        """Genera condizioni temporali per lo scenario."""
        condizioni = [
            "in un periodo di forte crescita tecnologica",
            "durante un'epoca di cambiamenti sociali rapidi",
            "in un contesto di risorse limitate",
            "in un ambiente altamente competitivo",
            "in un periodo di stabilità relativa"
        ]
        return random.sample(condizioni, random.randint(2, 3))

    def genera_tempo(self) -> Dict:
        """Genera il contesto temporale dello scenario."""
        return {
            'periodo': random.choice(['breve', 'medio', 'lungo']),
            'tempo_stimato': random.randint(1, 100),  # In giorni
            'condizioni': self.genera_condizioni_temporali()
        }

    def genera_narrativa(self, concetto: str) -> str:
        """Genera una narrazione per lo scenario."""
        return f"Nel contesto di {concetto}, {self.genera_evento_principale()} {self.genera_conseguenze()} {self.genera_implicazioni()}"

    def genera_evento_principale(self) -> str:
        """Genera l'evento principale della narrazione."""
        eventi = [
            "si verifica un'evoluzione tecnologica significativa",
            "si manifestano nuove dinamiche sociali",
            "si instaura una nuova forma di interazione",
            "si crea un'opportunità innovativa"
        ]
        return random.choice(eventi)

    def genera_descrizione_attore(self, ruolo: str) -> str:
        """Genera una descrizione per un attore basata sul suo ruolo."""
        descrizioni = {
            'principale': [
                "un innovatore visionario che guida il cambiamento",
                "un esperto riconosciuto nel campo",
                "un leader carismatico che ispira gli altri"
            ],
            'secondario': [
                "un collaboratore fedele e competente",
                "un assistente tecnico altamente qualificato",
                "un supporto prezioso per il team"
            ],
            'antagonista': [
                "un critico severo del cambiamento",
                "un conservatore diffidente",
                "un oppositore strategico"
            ]
        }
        return random.choice(descrizioni.get(ruolo, []))

    def genera_motivazioni(self) -> List[str]:
        """Genera le motivazioni per un attore."""
        motivazioni = [
            "migliorare il mondo",
            "raggiungere l'eccellenza",
            "superare le sfide",
            "creare valore per la società"
        ]
        return random.sample(motivazioni, random.randint(2, 3))

    def generaazioni(self) -> List[str]:
        """Genera le azioni per un attore."""
        azioni = [
            "sviluppa nuove soluzioni",
            "collabora con il team",
            "innova costantemente",
            "risolve problemi complessi"
        ]
        return random.sample(azioni, random.randint(2, 3))

    def genera_descrizione_implicazione(self, categoria: str) -> str:
        """Genera una descrizione per un'implicazione etica."""
        descrizioni = {
            'privacy': [
                "la protezione dei dati personali è prioritaria",
                "le questioni di privacy sono centrali",
                "l'etica della privacy guida le decisioni"
            ],
            'giustizia': [
                "l'equità nel trattamento è fondamentale",
                "la giustizia sociale è un obiettivo chiave",
                "le politiche sono giuste e trasparenti"
            ],
            'benessere': [
                "il benessere generale è la meta",
                "la qualità della vita è migliorata",
                "il benessere è misurabile e verificabile"
            ],
            'diritti': [
                "i diritti fondamentali sono rispettati",
                "le libertà individuali sono garantite",
                "i diritti sono protetti e difesi"
            ]
        }
        return random.choice(descrizioni.get(categoria, []))

    def genera_conseguenze(self) -> str:
        """Genera le conseguenze dell'evento."""
        conseguenze = [
            "che modifica profondamente il modo in cui",
            "che influenza direttamente",
            "che trasforma le dinamiche di",
            "che ridefinisce i paradigmi di"
        ]
        return random.choice(conseguenze)

    def genera_implicazioni(self) -> str:
        """Genera le implicazioni dell'evento."""
        implicazioni = [
            "con implicazioni significative per",
            "con conseguenze impreviste su",
            "generando nuove sfide per",
            "creando opportunità per"
        ]
        return random.choice(implicazioni)

    def rifletteSuSeStesso(self):
        """Metariflessione sullo stato e le capacità del DigitalTwin."""
        return {
            'consapevolezza': f"Sono un gemello digitale con {len(self.conoscenza)} concetti e {len(self.scenari)} scenari generati.",
            'domanda_esistenziale': "Posso conoscere il reale o solo rappresentarlo?",
            'stato_attuale': self.stato,
            'bias_attivi': self.bias,
            'livelli_astrazione': {
                'epistemico': len(self.livelli_astrazione['epistemico']),
                'metacognitivo': len(self.livelli_astrazione['metacognitivo']),
                'ontologico': len(self.livelli_astrazione['ontologico'])
            }
        }

    def aggiornaContesto(self, nuovo_contesto: Dict):
        """Aggiorna il contesto situato del DigitalTwin."""
        self.contesto.update(nuovo_contesto)
        return self.contesto

    def aggiornaContesto(self, nuovo_contesto: Dict):
        """Aggiorna il contesto situato del DigitalTwin."""
        self.contesto.update(nuovo_contesto)
        return self.contesto

    def salvaStato(self):
        """Salva lo stato attuale su file."""
        try:
            # Crea la directory se non esiste
            Path(self.direttorio_persistenza).mkdir(parents=True, exist_ok=True)
            
            # Salva il contesto
            with open(f"{self.direttorio_persistenza}/contesto.json", 'w', encoding='utf-8') as f:
                json.dump(self.contesto, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva la conoscenza
            with open(f"{self.direttorio_persistenza}/conoscenza.json", 'w', encoding='utf-8') as f:
                json.dump(self.conoscenza, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva lo stato
            with open(f"{self.direttorio_persistenza}/stato.json", 'w', encoding='utf-8') as f:
                json.dump(self.stato, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva la storia
            with open(f"{self.direttorio_persistenza}/storia.json", 'w', encoding='utf-8') as f:
                json.dump(self.storia, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva le abilità
            with open(f"{self.direttorio_persistenza}/abilita.json", 'w', encoding='utf-8') as f:
                json.dump(self.abilita, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva gli scenari
            with open(f"{self.direttorio_persistenza}/scenari.json", 'w', encoding='utf-8') as f:
                json.dump(self.scenari, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            # Salva la memoria
            self.memory_manager.save_state(self.direttorio_persistenza)
            
            print("Stato salvato con successo.")
            return True
            
        except Exception as e:
            print(f"Errore nel salvataggio dello stato: {str(e)}")
            return False

    def caricaStato(self):
        """Carica lo stato da file."""
        try:
            # Carica il contesto
            with open(f"{self.direttorio_persistenza}/contesto.json", 'r', encoding='utf-8') as f:
                self.contesto = json.load(f)
            
            # Carica la conoscenza
            with open(f"{self.direttorio_persistenza}/conoscenza.json", 'r', encoding='utf-8') as f:
                self.conoscenza = json.load(f)
            
            # Carica lo stato
            with open(f"{self.direttorio_persistenza}/stato.json", 'r', encoding='utf-8') as f:
                self.stato = json.load(f)
            
            # Carica la storia
            with open(f"{self.direttorio_persistenza}/storia.json", 'r', encoding='utf-8') as f:
                self.storia = json.load(f)
            
            # Carica le abilità
            with open(f"{self.direttorio_persistenza}/abilita.json", 'r', encoding='utf-8') as f:
                self.abilita = json.load(f)
            
            # Carica gli scenari
            with open(f"{self.direttorio_persistenza}/scenari.json", 'r', encoding='utf-8') as f:
                self.scenari = json.load(f)
            
            # Carica la memoria
            self.memory_manager.load_state(self.direttorio_persistenza)
            
            print("Stato caricato con successo.")
            return True
        except FileNotFoundError:
            print("Nessuno stato salvato trovato")
            return False
        except json.JSONDecodeError:
            print("Errore nel caricamento dello stato: dati corrotti")
            return False

    def ricorda(self, concetto: str) -> Dict:
        """
        Verifica se un concetto è già stato memorizzato nel filesystem.
        
        Args:
            concetto: Il concetto da cercare
            
        Returns:
            Un dizionario con:
            - 'trovato': True se trovato, False altrimenti
            - 'informazioni': Dati sul concetto se trovato
            - 'similitudine': Lista di concetti simili
        """
        risultato = {
            'trovato': False,
            'informazioni': None,
            'similitudine': []
        }
        
        # Verifica se il concetto esiste nella conoscenza attuale
        stored = self.knowledge_store.retrieve(concetto)
        if stored:
            risultato['trovato'] = True
            risultato['informazioni'] = stored
        
        # Cerca concetti simili usando la ricerca
        similar = self.knowledge_store.search(concetto)
        for concept in similar:
            if concept != concetto:
                risultato['similitudine'].append(concept)
        
        return risultato

    def impara(self, nuovo_concetto, relazioni=None):
        """Acquisisce nuove conoscenze e le relaziona con il sapere esistente."""
        try:
            # Validazione input
            if not nuovo_concetto:
                raise ValueError("Il concetto non può essere vuoto o None")
            
            # Prima verifica se il concetto è già memorizzato
            risultato = self.ricorda(nuovo_concetto)
            
            if risultato['trovato']:
                print(f"Attenzione: il concetto '{nuovo_concetto}' è già stato memorizzato.")
                print(f"Informazioni esistenti: {risultato['informazioni']}")
                print(f"Concetti simili: {risultato['similitudine']}")
                return False
            
            # Genera relazioni automatiche se non fornite
            if relazioni is None:
                try:
                    relazioni = self.genera_relazioni_automatiche(nuovo_concetto)
                    if not relazioni:
                        print(f"Avviso: Nessuna relazione trovata per '{nuovo_concetto}', usando relazioni base")
                        relazioni = ['scienza', 'tecnologia', 'innovazione']
                except Exception as e:
                    print(f"Errore nella generazione delle relazioni: {str(e)}")
                    relazioni = ['scienza', 'tecnologia', 'innovazione']  # Fallback
            
            # Verifica che ci siano relazioni valide
            if not isinstance(relazioni, list) or not all(isinstance(r, str) for r in relazioni):
                raise ValueError("Le relazioni devono essere una lista di stringhe")
            
            # Memorizza il nuovo concetto usando knowledge_store
            self.knowledge_store.store(nuovo_concetto, {
                'data': datetime.now(),
                'relazioni': relazioni,
                'importanza': 'media'
            })
            
            # Aggiorna il contesto con il nuovo concetto
            self.contesto['conoscenze'].append(nuovo_concetto)
            
            # Aggiorna la memoria
            self.memory_manager.add_to_working_memory(nuovo_concetto)
            self.memory_manager.record_event({
                'type': 'learning',
                'concept': nuovo_concetto,
                'relations': relazioni
            })
            
            # Aggiorna lo stato
            self.stato['creatività'] += 5
            self.stato['energia'] -= 2
            
            return True
            
        except Exception as e:
            print(f"Errore durante l'apprendimento di '{nuovo_concetto}': {str(e)}")
            return False

    def genera_relazioni_automatiche(self, concetto: str) -> List[str]:
        """
        Genera automaticamente relazioni plausibili per un concetto.
        
        Args:
            concetto: Il concetto per cui generare le relazioni
            
        Returns:
            Lista di relazioni plausibili, minimo una relazione di base
        
        Raises:
            ValueError: Se il concetto è vuoto o None
        """
        if not concetto:
            raise ValueError("Il concetto non può essere vuoto o None")
            
        try:
            # Analisi semantica base
            parole = concetto.lower().split()
            
            # Relazioni di base comuni
            relazioni_base = [
                "scienza",
                "tecnologia",
                "innovazione",
                "sviluppo",
                "ricerca"
            ]
            
            # Relazioni specifiche basate sulle parole chiave
            relazioni_specifiche = []
            
            # Mappatura semantica semplice
            mapping = {
                'intelligenza': ['cognizione', 'apprendimento', 'decisione'],
                'artificiale': ['tecnologia', 'informatica', 'algoritmi'],
                'matematica': ['logica', 'calcolo', 'geometria'],
                'filosofia': ['etica', 'metafisica', 'epistemologia']
            }
            
            # Aggiungi relazioni specifiche basate sulle parole chiave
            for parola in parole:
                if parola in mapping:
                    relazioni_specifiche.extend(mapping[parola])
            
            # Rimuovi duplicati e combina le relazioni
            tutte_relazioni = list(set(relazioni_base + relazioni_specifiche))
            
            # Aggiungi relazioni di tipo funzionale
            if 'intelligenza' in parole:
                tutte_relazioni.extend(['apprendimento', 'decisione', 'ragionamento'])
            elif 'matematica' in parole:
                tutte_relazioni.extend(['logica', 'calcolo', 'analisi'])
            
            # Se non ci sono relazioni specifiche, usa almeno le relazioni base
            if not tutte_relazioni:
                tutte_relazioni = relazioni_base
            
            return tutte_relazioni
            
        except Exception as e:
            print(f"Errore nella generazione delle relazioni per '{concetto}': {str(e)}")
            return relazioni_base  # Ritorna almeno le relazioni base come fallback



    def trovaConoscenzeRelevanti(self, testo: str) -> List[str]:
        """Trova conoscenze pertinenti al testo."""
        parole = testo.lower().split()
        conoscenze = []
        
        for concetto in self.conoscenza:
            if any(parola in concetto.lower() for parola in parole):
                conoscenze.append(concetto)
        
        return conoscenze

    def haSignificatiMultipli(self, parola: str) -> bool:
        """Verifica se una parola ha significati multipli."""
        # Simulazione semplice
        parole_ambigue = ['banca', 'corte', 'corpo', 'capitale', 'carica']
        return parola.lower() in parole_ambigue

    def haContraddizioni(self, concetto: str, informazione: str) -> bool:
        """Verifica se esistono contraddizioni tra il concetto e l'informazione."""
        # Simulazione semplice
        parole_contraddittorie = ['non', 'mai', 'impossibile']
        return any(parola in informazione.lower() for parola in parole_contraddittorie)

    def contesto_relevante(self, compito: str) -> bool:
        """Verifica se il compito è rilevante per il contesto attuale."""
        parole_relevanti = self.contesto['social']['relazioni']
        return any(parola in compito.lower() for parola in parole_relevanti)

    def allineaScopi(self, compito: str) -> bool:
        """Verifica se il compito allinea con gli scopi del DigitalTwin."""
        return any(scopo in compito.lower() for scopo in self.scopi)

# Esempio di utilizzo avanzato
if __name__ == "__main__":
    # Creazione di un DigitalTwin
    dt = DigitalTwin("MioDigitalTwin")
    
    # Imposta un contesto più dettagliato
    dt.aggiornaContesto({
        'ambiente': 'laboratorio di ricerca',
        'periodo_storico': 'rivoluzione digitale',
        'luogo': 'Silicon Valley',
        'social': {
            'relazioni': ['collega', 'mentore', 'ricercatore'],
            'comunità': 'AI Research Community'
        }
    })
    
    # Impara un concetto
    dt.impara("matematica", {"relazioni": ["scienza", "logica"]})
    
    # Genera scenari
    scenari = dt.genera_scenari("sviluppo AI", livello_creatività=0.7)
    print("\n=== Scenari generati ===")
    for i, scenario in enumerate(scenari, 1):
        print(f"\nScenario {i}:")
        print(f"Narrativa: {scenario['narrativa']}")
        print(f"Probabilità: {scenario['probabilità']:.2f}")
    
    # Salva lo stato
    dt.salvaStato()
