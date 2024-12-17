import os

def create_checkpoint_structure(base_path):
    """Crea la estructura inicial de carpetas para checkpoints"""
    checkpoint_path = os.path.join(base_path, 'checkpoints')
    
    # Crear directorio principal de checkpoints
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Crear subdirectorios para cada modelo
    model_dirs = [
        'RegresionPolinomica',
        'RegresionLineal',
        'SeriesTemporales',
        'DeteccionAnomalias'
    ]
    
    for model_dir in model_dirs:
        model_path = os.path.join(checkpoint_path, model_dir)
        os.makedirs(model_path, exist_ok=True)
        
    return checkpoint_path
