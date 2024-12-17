import os
import pandas as pd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

## Función para cargar los datos desde todas las subcarpetas dentro de Forex
"""
def load_data_from_forex(base_folder):
    data_files = []
    
    # Recorremos las subcarpetas dentro de la carpeta Forex
    forex_folder = os.path.join(base_folder, 'Forex')
    for subfolder in os.listdir(forex_folder):
        subfolder_path = os.path.join(forex_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Si es una carpeta, buscamos archivos .csv dentro de ella
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    data_files.append(os.path.join(subfolder_path, file))
    
    return data_files
"""
    ##
# Función para procesar el DataFrame y extraer picos y valles
def process_data(df):
    picos = []
    valles = []
    angles = []
    
    for i in range(1, len(df) - 1):
        if df['Close'][i] > df['Close'][i - 1] and df['Close'][i] > df['Close'][i + 1]:  # Pico
            picos.append((df['Date'][i], df['Close'][i]))
            if i > 1 and i < len(df) - 2:
                # Calculamos los ángulos entre los picos y los valles cercanos
                angle_prev = calculate_angle((df['Date'][i-1], df['Close'][i-1]), (df['Date'][i], df['Close'][i]))
                angle_next = calculate_angle((df['Date'][i], df['Close'][i]), (df['Date'][i+1], df['Close'][i+1]))
                angles.append((df['Date'][i], angle_prev, angle_next))
        
        elif df['Close'][i] < df['Close'][i - 1] and df['Close'][i] < df['Close'][i + 1]:  # Valle
            valles.append((df['Date'][i], df['Close'][i]))
            if i > 1 and i < len(df) - 2:
                # Calculamos los ángulos entre los picos y los valles cercanos
                angle_prev = calculate_angle((df['Date'][i-1], df['Close'][i-1]), (df['Date'][i], df['Close'][i]))
                angle_next = calculate_angle((df['Date'][i], df['Close'][i]), (df['Date'][i+1], df['Close'][i+1]))
                angles.append((df['Date'][i], angle_prev, angle_next))
    
    return picos, valles, angles

# Función para calcular el ángulo entre dos puntos (para los picos y valles)
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = np.arctan(delta_y / delta_x) * (180 / np.pi)
    return angle

# Función para mostrar los resultados (picos, valles y tendencias)
def plot_trends(df, picos, valles):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    
    # Plotting picos and valles
    picos_dates = [p[0] for p in picos]
    picos_values = [p[1] for p in picos]
    valles_dates = [v[0] for v in valles]
    valles_values = [v[1] for v in valles]
    
    plt.scatter(picos_dates, picos_values, color='red', label='Picos')
    plt.scatter(valles_dates, valles_values, color='green', label='Valles')
    
    plt.title('Picos y Valles')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.legend()
    plt.show()

# Función para preparar los datos para el entrenamiento del modelo
def prepare_data(df, angles):
    X = []
    y = []
    
    for i in range(len(df) - 1):
        trend = df['Close'][i+1] - df['Close'][i]
        
        if trend > 0:
            # Aquí estamos agregando el ángulo y el cambio de tendencia
            X.append([df['Close'][i], df['Open'][i], angles[i][1], angles[i][2]])  # Incluye ángulos de picos/valles
            y.append(1)  # 1 para tendencia alcista
        else:
            X.append([df['Close'][i], df['Open'][i], angles[i][1], angles[i][2]])  # Incluye ángulos de picos/valles
            y.append(0)  # 0 para tendencia bajista
    
    return np.array(X), np.array(y)

# Función para entrenar el modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalización de los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicciones y evaluación
    y_pred = model.predict(X_test)
    print("Clasificación de Resultados: \n", classification_report(y_test, y_pred))
    
    return model, scaler

# Función principal
"""
#def main():
    # Ruta de la carpeta base
    base_folder = 'C:/Users/jeron/Desktop/contenedor/pro_1/database'
    
    # Cargar todos los archivos de datos desde Forex y sus subcarpetas
    data_files = load_data_from_forex(base_folder)
    
    for file in data_files:
        print(f"Cargando archivo {file}")
        
        # Cargar el archivo CSV
        df = pd.read_csv(file)
        
        # Procesar los datos
        picos, valles, angles = process_data(df)
        
        # Mostrar los resultados
        plot_trends(df, picos, valles)
        
        # Preparar los datos para el modelo
        X, y = prepare_data(df, angles)
        
        # Entrenar el modelo
        model, scaler = train_model(X, y)
        
        # Aquí se puede guardar el modelo para su uso posterior
        # Ejemplo: joblib.dump(model, 'modelo_forex.pkl')
        
        print(f"Modelo entrenado con datos de {file}")

if __name__ == '__main__':
    main()
"""
def load_data_from_forex(base_folder):
    forex_folder = os.path.join(base_folder, 'forex')  # Asegurándonos de usar barras normales
    
    if not os.path.exists(forex_folder):
        print(f"¡La ruta no existe! Verifica que la carpeta 'forex' esté en {base_folder}")
        return []
    
    data_files = []
    
    # Recorremos las subcarpetas dentro de la carpeta forex
    for subfolder in os.listdir(forex_folder):
        subfolder_path = os.path.join(forex_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Solo si es una carpeta
            print(f"Accediendo a la carpeta: {subfolder_path}")
            
            # Buscar archivos Excel dentro de la subcarpeta
            for file in os.listdir(subfolder_path):
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    file_path = os.path.join(subfolder_path, file)
                    data_files.append(file_path)
    
    return data_files

def main():
    base_folder = "C:/Users/jeron/Desktop/contenedor/pro_1/database"
    
    # Cargar los archivos Excel desde Forex
    data_files = load_data_from_forex(base_folder)
    
    # Cargar y procesar cada archivo Excel
    for file_path in data_files:
        print(f"Cargando archivo: {file_path}")
        df = pd.read_excel(file_path)
        # Procesar los datos cargados
        print(df.head())  # Muestra las primeras filas del DataFrame cargado

if __name__ == '__main__':
    main()
