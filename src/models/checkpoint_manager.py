import os
import json
import pickle
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.checkpoints_dir = os.path.join(base_path, 'checkpoints')
        self.create_checkpoint_structure()
        
    def create_checkpoint_structure(self):
        models = ['RegresionPolinomica', 'RegresionLineal', 
                 'SeriesTemporales', 'DeteccionAnomalias']
        for model in models:
            model_path = os.path.join(self.checkpoints_dir, model)
            os.makedirs(model_path, exist_ok=True)
    
    def save_checkpoint(self, model_name, model, performance_metrics, version):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            model_name, 
            f'checkpoint_v{version}_{timestamp}'
        )
        
        with open(f'{checkpoint_path}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        metrics_data = {
            'version': version,
            'timestamp': timestamp,
            'metrics': performance_metrics,
            'hyperparameters': model.get_params()
        }
        
        with open(f'{checkpoint_path}_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
        logger.info(f"Checkpoint saved for {model_name} with version {version}")

    def should_save_checkpoint(self, model_name, current_performance):
        previous_performance = self.get_previous_performance(model_name)
        if previous_performance is None or current_performance < previous_performance:
            return True
        return False

    def get_previous_performance(self, model_name):
        model_path = os.path.join(self.checkpoints_dir, model_name)
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('_metrics.json')]
        if not checkpoint_files:
            return None
        
        last_checkpoint = max(checkpoint_files)
        with open(os.path.join(model_path, last_checkpoint), 'r') as f:
            data = json.load(f)
        return data['metrics']['performance']

    def load_best_checkpoint(self, model_name):
        """Carga el mejor checkpoint basado en métricas"""
        model_path = os.path.join(self.checkpoints_dir, model_name)
        best_performance = float('-inf')
        best_checkpoint = None
        
        for file in os.listdir(model_path):
            if file.endswith('_metrics.json'):
                with open(os.path.join(model_path, file), 'r') as f:
                    metrics = json.load(f)
                    if metrics['metrics']['performance'] > best_performance:
                        best_performance = metrics['metrics']['performance']
                        best_checkpoint = file.replace('_metrics.json', '.pkl')
        
        if best_checkpoint:
            with open(os.path.join(model_path, best_checkpoint), 'rb') as f:
                return pickle.load(f)
        return None