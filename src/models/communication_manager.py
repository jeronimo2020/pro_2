from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class ModelResult:
    predictions: np.ndarray
    confidence: float
    metadata: Dict[str, Any]

class AdministradorDeComunicacion:
    def __init__(self):
        self.model_results: Dict[str, ModelResult] = {}
        self.model_status = {}
        
    def register_result(self, model_name: str, result: ModelResult):
        self.model_results[model_name] = result
        self.validate_coherence()
        
    def validate_coherence(self):
        """Validate results coherence across models"""
        pass

    def get_consolidated_results(self):
        return self.model_results
