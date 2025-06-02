import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from LLMManager import LLMManager

class MemoryManager:
    def __init__(self, llm_manager: LLMManager):
        """Inizializza il sistema di memoria."""
        self.llm_manager = llm_manager
        
        # Working Memory (Breve Termine)
        self.working_memory = {
            'attivo': set(),  # Concetti attualmente attivi
            'timestamp': datetime.now(),
            'capacita': 7,   # Limite di Miller
            'priorita': []   # Stack di priorità
        }
        
        # Memoria Episodica (Lungo Termine)
        self.episodic_memory = {
            'eventi': [],
            'interazioni': [],
            'timestamp': datetime.now()
        }
        
        # Cache per i vettori delle memorie episodiche
        self.embedding_cache = {}
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Genera un embedding per il testo."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.llm_manager.generate_embedding(text)
        self.embedding_cache[text] = embedding
        return embedding
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcola la similarità coseno tra due vettori."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def add_to_working_memory(self, concept: str, priority: int = 0) -> bool:
        """Aggiunge un concetto alla Working Memory."""
        if concept in self.working_memory['attivo']:
            return False
            
        if len(self.working_memory['attivo']) >= self.working_memory['capacita']:
            # Rimuovi il meno prioritario
            self.working_memory['attivo'].remove(self.working_memory['priorita'].pop(0))
            
        self.working_memory['attivo'].add(concept)
        self.working_memory['priorita'].append((priority, concept))
        self.working_memory['priorita'].sort(reverse=True)  # Priorità più alta prima
        self.working_memory['timestamp'] = datetime.now()
        return True
    
    def get_working_memory(self) -> List[str]:
        """Restituisce i concetti attualmente nella Working Memory."""
        return list(self.working_memory['attivo'])
    
    def record_event(self, event: Dict):
        """Registra un evento nella memoria episodica."""
        event['timestamp'] = datetime.now()
        self.episodic_memory['eventi'].append(event)
        self.episodic_memory['timestamp'] = datetime.now()
    
    def record_interaction(self, interaction: Dict):
        """Registra un'interazione nella memoria episodica."""
        interaction['timestamp'] = datetime.now()
        self.episodic_memory['interazioni'].append(interaction)
        self.episodic_memory['timestamp'] = datetime.now()
    
    def search_episodic_memory(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """Cerca eventi simili nella memoria episodica."""
        query_embedding = self._generate_embedding(query)
        results = []
        
        for event in self.episodic_memory['eventi']:
            if 'description' in event:
                similarity = self._calculate_similarity(
                    query_embedding,
                    self._generate_embedding(event['description'])
                )
                if similarity > threshold:
                    results.append({
                        'event': event,
                        'similarity': similarity
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return [r['event'] for r in results]
    
    def get_recent_interactions(self, n: int = 5) -> List[Dict]:
        """Restituisce le n interazioni più recenti."""
        return self.episodic_memory['interazioni'][-n:]
    
    def clear_working_memory(self):
        """Pulisce la Working Memory."""
        self.working_memory['attivo'].clear()
        self.working_memory['priorita'].clear()
        self.working_memory['timestamp'] = datetime.now()
    
    def save_state(self, directory: str):
        """Salva lo stato della memoria."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Salva Working Memory
        with open(f"{directory}/working_memory.json", 'w', encoding='utf-8') as f:
            json.dump({
                'attivo': list(self.working_memory['attivo']),
                'priorita': self.working_memory['priorita'],
                'timestamp': self.working_memory['timestamp'].isoformat()
            }, f, ensure_ascii=False)
        
        # Salva Episodic Memory
        with open(f"{directory}/episodic_memory.json", 'w', encoding='utf-8') as f:
            json.dump({
                'eventi': self.episodic_memory['eventi'],
                'interazioni': self.episodic_memory['interazioni'],
                'timestamp': self.episodic_memory['timestamp'].isoformat()
            }, f, ensure_ascii=False)
    
    def load_state(self, directory: str):
        """Carica lo stato della memoria."""
        try:
            # Carica Working Memory
            with open(f"{directory}/working_memory.json", 'r', encoding='utf-8') as f:
                wm_data = json.load(f)
                self.working_memory['attivo'] = set(wm_data['attivo'])
                self.working_memory['priorita'] = wm_data['priorita']
                self.working_memory['timestamp'] = datetime.fromisoformat(wm_data['timestamp'])
            
            # Carica Episodic Memory
            with open(f"{directory}/episodic_memory.json", 'r', encoding='utf-8') as f:
                em_data = json.load(f)
                self.episodic_memory['eventi'] = em_data['eventi']
                self.episodic_memory['interazioni'] = em_data['interazioni']
                self.episodic_memory['timestamp'] = datetime.fromisoformat(em_data['timestamp'])
            
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Errore nel caricamento della memoria: {str(e)}")
            return False
