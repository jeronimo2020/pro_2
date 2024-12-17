from abc import ABC, abstractmethod
import numpy as np
class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.version = 1.0
        self.performance_history = []

    @abstractmethod
    def train(self, data: np.ndarray) -> float:
        """Train the model and return performance metric"""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def save_version(self):
        """Save model version if performance improves"""
        pass
