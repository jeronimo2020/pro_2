from src.models.communication_manager import AdministradorDeComunicacion
from src.models.peripheral_models import *
from src.data.data_loader import load_all_excel_files
from src.training.trainer import ModelTrainer
from src.utils.checkpoint_setup import create_checkpoint_structure
import os

def main():
    # Ruta base del proyecto
    base_path = r"C:\Users\jeron\Desktop\contenedor\pro_2"
    
    # Crear estructura de checkpoints
    checkpoint_path = create_checkpoint_structure(base_path)
    print(f"Estructura de checkpoints creada en: {checkpoint_path}")
    
    # Initialize components
    comm_manager = AdministradorDeComunicacion()
    
    # Load data from forex directory
    forex_path = os.path.join(base_path, "database", "forex")
    all_forex_data = load_all_excel_files(forex_path)
    
    # Initialize models
    models = {
        "RegresionPolinomica": RegresionPolinomica(),
        "RegresionLineal": RegresionLineal(),
        "SeriesTemporales": SeriesTemporales(),
        "DeteccionAnomalias": DeteccionAnomalias()
    }
    
    # Initialize trainer with base_path
    trainer = ModelTrainer(models, base_path)
    
    # Proceso de entrenamiento que guardará checkpoints
    for file_path, df in all_forex_data.items():
        print(f"Procesando: {file_path}")
        windows = trainer.create_sliding_windows(df, window_size=5)
        for window in windows:
            results = trainer.train_all_models(window)
if __name__ == "__main__":
    main()