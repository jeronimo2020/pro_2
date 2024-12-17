import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    EXPECTED_COLUMNS = ['timestamp', 'volume', 'open', 'max', 'min', 'close']

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_excel_file(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_excel(file_path)
        self.validate_columns(df, file_path)
        return df
        
    def validate_columns(self, df: pd.DataFrame, file_path: Path):
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns in {file_path}: {missing_cols}")
            raise ValueError(f"Required columns missing: {missing_cols}")
    
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
            excel_files[relative_path] = self.load_excel_file(excel_file)

        return excel_files
    def create_sliding_windows(self, df: pd.DataFrame, window_size: int):
        return [df.iloc[i:i + window_size] for i in range(len(df) - window_size + 1)]

# Usage example:
forex_path = r"C:\Users\jeron\Desktop\contenedor\pro_2\database\forex"
data_loader = DataLoader(forex_path)
all_excel_data = data_loader.load_all_excel_files()
