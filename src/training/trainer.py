from joblib import Parallel, delayed
import logging
import pandas as pd
from src.models.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models, base_path):
        self.models = models
        self.checkpoint_manager = CheckpointManager(base_path)
        self.current_versions = {model_name: 1.0 for model_name in models.keys()}
        
    def train_model(self, model_name, model, data):
        logger.info(f"Training model: {model_name}")
        performance = model.train(data)
        return model_name, performance
        
    def train_all_models(self, data):
        logger.info("Starting parallel training for all models")
        results = Parallel(n_jobs=-1)(
            delayed(self.train_model)(name, model, data) 
            for name, model in self.models.items()
        )
        
        processed_results = {}
        for model_name, performance in results:
            if self.checkpoint_manager.should_save_checkpoint(model_name, performance):
                self.checkpoint_manager.save_checkpoint(
                    model_name,
                    self.models[model_name],
                    {'performance': performance},
                    self.current_versions[model_name]
                )
                self.current_versions[model_name] += 0.1
            processed_results[model_name] = performance
            
        return processed_results

    def create_sliding_windows(self, data: pd.DataFrame, window_size: int):
        return [data.iloc[i:i + window_size] for i in range(len(data) - window_size + 1)]