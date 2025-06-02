import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from DigitalTwin import DigitalTwin
import os
import json
from datetime import datetime

class DigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Twin - Interfaccia Grafica")
        self.root.geometry("1200x800")
        
        # Inizializza il DigitalTwin
        self.dt = DigitalTwin("DigitalTwinGUI")
        
        # Crea la struttura dell'interfaccia
        self.crea_interfaccia()
        
    def crea_interfaccia(self):
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame per la sezione di input
        input_frame = ttk.LabelFrame(main_frame, text="Interazione", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Campo per il nome del DigitalTwin
        ttk.Label(input_frame, text="Nome DigitalTwin:").grid(row=0, column=0, sticky=tk.W)
        self.nome_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.nome_var).grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Campo per il concetto
        ttk.Label(input_frame, text="Concetto:").grid(row=1, column=0, sticky=tk.W)
        self.concetto_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.concetto_var).grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Campo per le relazioni
        ttk.Label(input_frame, text="Relazioni:").grid(row=2, column=0, sticky=tk.W)
        self.relazioni_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.relazioni_var).grid(row=2, column=1, sticky=(tk.W, tk.E))
        
        # Slider per il livello di creatività
        ttk.Label(input_frame, text="Livello di Creatività:").grid(row=3, column=0, sticky=tk.W)
        self.creatività_var = tk.DoubleVar(value=0.7)
        ttk.Scale(input_frame, from_=0, to=1, variable=self.creatività_var, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=(tk.W, tk.E))
        
        # Pulsanti
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Impara", command=self.impara_concetto).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Genera Scenari", command=self.genera_scenari).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Transizione", command=self.transizione).grid(row=0, column=2, padx=5)
        
        # Frame per i risultati
        self.risultati_frame = ttk.LabelFrame(main_frame, text="Risultati", padding="10")
        self.risultati_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Area di testo per i risultati
        self.risultati_text = scrolledtext.ScrolledText(self.risultati_frame, width=100, height=20)
        self.risultati_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configura il grid per espandere correttamente
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.risultati_frame.columnconfigure(0, weight=1)
        self.risultati_frame.rowconfigure(0, weight=1)
        
    def impara_concetto(self):
        try:
            nome = self.nome_var.get()
            concetto = self.concetto_var.get()
            relazioni = self.relazioni_var.get()
            
            if not nome or not concetto:
                messagebox.showerror("Errore", "Nome e concetto sono campi obbligatori")
                return
                
            # Aggiorna il nome del DigitalTwin
            self.dt.nome = nome
            
            # Converte le relazioni da stringa a lista
            relazioni_lista = [r.strip() for r in relazioni.split(',') if r.strip()]
            
            # Impara il concetto
            self.dt.impara(concetto, {"relazioni": relazioni_lista})
            
            self.aggiorna_risultati(f"Concetto '{concetto}' imparato con successo\n")
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            
    def genera_scenari(self):
        try:
            concetto = self.concetto_var.get()
            creatività = self.creatività_var.get()
            
            if not concetto:
                messagebox.showerror("Errore", "Il concetto è obbligatorio")
                return
                
            scenari = self.dt.genera_scenari(concetto, livello_creatività=creatività)
            
            self.aggiorna_risultati("\n=== Scenari generati ===\n")
            for i, scenario in enumerate(scenari, 1):
                self.aggiorna_risultati(f"\nScenario {i}:\n")
                self.aggiorna_risultati(f"Narrativa: {scenario['narrativa']}\n")
                self.aggiorna_risultati(f"Probabilità: {scenario['probabilità']:.2f}\n")
                self.aggiorna_risultati("Attori principali:\n")
                for attore in scenario['attori']:
                    self.aggiorna_risultati(f"- {attore['ruolo']}: {attore['descrizione']}\n")
                self.aggiorna_risultati("Implicazioni etiche:\n")
                for implicazione in scenario['implicazioni_etiche']:
                    self.aggiorna_risultati(f"- {implicazione}\n")
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            
    def transizione(self):
        try:
            domanda = self.concetto_var.get()
            if not domanda:
                messagebox.showerror("Errore", "Inserisci una domanda critica")
                return
                
            if self.dt.transizioneLivelli("ontologico", "epistemico", domanda):
                self.aggiorna_risultati("\nTransizione avvenuta con successo\n")
            else:
                self.aggiorna_risultati("\nTransizione non riuscita\n")
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            
    def aggiorna_risultati(self, testo):
        self.risultati_text.insert(tk.END, testo)
        self.risultati_text.see(tk.END)
        
    def salva_stato(self):
        try:
            self.dt.salvaStato()
            self.aggiorna_risultati("\nStato salvato con successo\n")
        except Exception as e:
            messagebox.showerror("Errore", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitalTwinGUI(root)
    root.mainloop()
