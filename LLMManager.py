import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LLMManager:
    def __init__(self):
        """Inizializza il gestore LLM."""
        self.model = None
        self.context = {}
        
    def generate_response(self, prompt: str) -> str:
        """Genera una risposta basata sul prompt."""
        # Placeholder per l'implementazione del modello LLM
        return f"Risposta generata per: {prompt}"
        
    def update_context(self, new_context: dict):
        """Aggiorna il contesto del modello."""
        self.context.update(new_context)
        
    def get_context(self) -> dict:
        """Restituisce il contesto corrente."""
        return self.context
