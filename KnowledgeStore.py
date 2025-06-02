import sys
import os
from datetime import datetime
from typing import Dict, Any, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLMManager import LLMManager

class KnowledgeStore:
    def __init__(self, llm_manager):
        """Inizializza il sistema di archiviazione delle conoscenze."""
        self.llm_manager = llm_manager
        self.knowledge = {}
        self.context = {}
        
    def store(self, concept: str, information: Dict):
        """Archivia una nuova conoscenza con i suoi bias associati."""
        self.knowledge[concept] = {
            'valore': information.get('valore', 1.0),
            'relazioni': information.get('relazioni', []),
            'bias': information.get('bias', {
                'conferma': 0.0,
                'disponibilità': 0.0,
                'ancoraggio': 0.0,
                'loss_aversion': 0.0
            }),
            'timestamp': datetime.now()
        }
        
    def retrieve(self, concept: str) -> Dict:
        """Recupera una conoscenza con i suoi bias."""
        return self.knowledge.get(concept)
        
    def get_concept_value(self, concept: str) -> float:
        """Restituisce il valore di un concetto."""
        info = self.retrieve(concept)
        return info['valore'] if info else 1.0
        
    def update_context(self, context: Dict):
        """Aggiorna il contesto delle conoscenze."""
        self.context.update(context)
        
    def get_context(self) -> Dict:
        """Restituisce il contesto corrente."""
        return self.context
        
    def search(self, query: str) -> Dict:
        """Cerca conoscenze correlate alla query."""
        results = {}
        for concept, info in self.knowledge.items():
            if query.lower() in concept.lower():
                results[concept] = info
        return results
        
    def find_contraddizioni(self, concept: str) -> List[str]:
        """Trova concetti che potrebbero contraddire il nuovo concetto."""
        contraddizioni = []
        for existing_concept, info in self.knowledge.items():
            # Se il concetto esistente ha un valore alto e bias di conferma basso
            if info['valore'] > 0.8 and info['bias']['conferma'] < 0.3:
                contraddizioni.append(existing_concept)
        return contraddizioni
        
    def get_concepts_by_bias(self, bias_type: str, threshold: float = 0.5) -> List[str]:
        """Restituisce i concetti con un certo tipo di bias sopra una soglia."""
        return [
            concept for concept, info in self.knowledge.items()
            if info['bias'].get(bias_type, 0.0) >= threshold
        ]
