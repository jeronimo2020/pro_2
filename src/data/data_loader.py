import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ForexDataLoader:
    def __init__(self):
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.data_cache = {}
        
    def load_and_validate(self, file_path: str) -> Optional[pd.DataFrame]:
        """Carga y valida el formato de datos forex"""
        try:
            df = pd.read_excel(file_path)
            if not all(col in df.columns for col in self.required_columns):
                raise ValueError(f"Columnas requeridas faltantes en {file_path}")
                
            # Preprocesamiento
            df['DateTime'] = pd.to_datetime(df.index)
            df['Hour'] = df['DateTime'].dt.hour
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            
            # Cálculos técnicos básicos
            df['TR'] = self.calculate_true_range(df)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
            
    @staticmethod
    def calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """Calcula el True Range para análisis de volatilidad"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range

    def load_all_excel_files(self):
        """
        Carga todos los archivos Excel de un directorio en un diccionario de DataFrames

        Returns:
            dict: Diccionario con {nombre_archivo: DataFrame}
        """
        excel_files = {}

        for excel_file in self.data_path.glob('**/*.xlsx'):
            relative_path = str(excel_file.relative_to(self.data_path))
            logger.info(f"Processing: {relative_path}")
            excel_files[relative_path] = self.load_and_validate(str(excel_file))

        return excel_files

    def create_sliding_windows(self, df: pd.DataFrame, window_size: int):
        return [df.iloc[i:i + window_size] for i in range(len(df) - window_size + 1)]

# Usage example:
forex_path = r"C:\Users\jeron\Desktop\contenedor\pro_2\database\forex"
data_loader = ForexDataLoader()
all_excel_data = data_loader.load_all_excel_files()
