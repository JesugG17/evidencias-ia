import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
data = pd.read_csv("C:\\Users\\alanm\\Desktop\\Inteligencia artificial\\Practicas\\Ultima unidad\\archivo_normalizado.csv")

# Definir el tamaño de la ventana de tiempo
window_size = 30

# Dividir los datos en conjuntos de entrenamiento y validación
train_data, val_data = train_test_split(data, test_size=0.3, shuffle=False)

# Construir los datos de entrenamiento
X_train = []
y_train = []
for i in range(len(train_data) - window_size):
    X_train.append(train_data.iloc[i:i+window_size].values)  # Tomar la ventana de tiempo como características
    y_train.append(train_data.iloc[i + window_size]['Agua'])  # Tomar el valor objetivo después de la ventana de tiempo
X_train = np.array(X_train)
y_train = np.array(y_train)

# Construir los datos de validación
X_val = []
y_val = []
for i in range(len(val_data) - window_size):
    X_val.append(val_data.iloc[i:i+window_size].values)  # Tomar la ventana de tiempo como características
    y_val.append(val_data.iloc[i + window_size]['Agua'])  # Tomar el valor objetivo después de la ventana de tiempo
X_val = np.array(X_val)
y_val = np.array(y_val)

# Definir el modelo con múltiples capas LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, X_train.shape[2])),  # Primera capa LSTM con return_sequences=True
    tf.keras.layers.LSTM(64, activation='relu'),  # Segunda capa LSTM
    tf.keras.layers.Dense(64, activation='relu'),  # Capa densa con 64 neuronas
    tf.keras.layers.Dense(1)  # Capa de salida con una sola neurona para la predicción
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Realizar predicciones en el conjunto de validación
y_pred = model.predict(X_val).flatten()  # Aplanar las predicciones a un arreglo unidimensional

# Calcular el Error Cuadrático Medio (MSE) en el conjunto de validación
mse = mean_squared_error(y_val, y_pred)
print("MSE del conjunto de validación:", mse)

# Mostrar la gráfica del pronóstico
plt.figure(figsize=(10, 6))
plt.plot(val_data.index[window_size:], y_val, label="Datos reales")
plt.plot(val_data.index[window_size:], y_pred, label="Pronóstico", linestyle='--')
plt.xlabel("Tiempo")
plt.ylabel("Demanda de agua")
plt.title("Pronóstico de la demanda de agua")
plt.legend()
plt.show()