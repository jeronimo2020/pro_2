from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd

@dataclass
class ModelMessage:
    sender: str
    data: pd.DataFrame
    patterns: Dict
    confidence: float
    timestamp: float

class AdministradorDeComunicacion:
    def __init__(self):
        self.message_queue = []
        self.model_states = {}
        self.column_mapping = {
            'Open': 'precio_apertura',
            'High': 'precio_maximo',
            'Low': 'precio_minimo',
            'Close': 'precio_cierre',
            'Volume': 'volumen'
        }
        
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza los nombres de columnas y formatos"""
        standardized = df.copy()
        standardized.columns = [self.column_mapping.get(col, col) for col in df.columns]
        return standardized
        
    def broadcast_pattern(self, sender: str, pattern_data: Dict):
        """Transmite patrones identificados a todos los modelos"""
        message = ModelMessage(
            sender=sender,
            data=pattern_data['data'],
            patterns=pattern_data['patterns'],
            confidence=pattern_data['confidence'],
            timestamp=np.datetime64('now')
        )
        self.message_queue.append(message)
        
    def get_model_updates(self, model_id: str) -> List[ModelMessage]:
        """Recupera actualizaciones relevantes para un modelo específico"""
        return [msg for msg in self.message_queue if msg.sender != model_id]