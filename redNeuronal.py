import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyswarm import pso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos CSV
data = pd.read_csv('C:\\Users\\alanm\\Downloads\\entrada y salida normalizado.csv')

# Dividir los datos en características (X) y salida esperada (y)
X = data.iloc[:, :5].values
y = data.iloc[:, 9].values

# Normalizar cada columna en el rango de 0 a 1
def min_max_normalize(column):
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

X_normalized = np.apply_along_axis(min_max_normalize, axis=0, arr=X)
y_normalized = min_max_normalize(y)

# Concatenar X y y para facilitar la eliminación de outliers
all_data = np.concatenate((X_normalized, y_normalized.reshape(-1, 1)), axis=1)

# Eliminar outliers usando el método de los cuartiles (IQR) después de la normalización
Q1 = np.percentile(all_data, 25, axis=0)
Q3 = np.percentile(all_data, 75, axis=0)
IQR = Q3 - Q1
outlier_mask = (all_data >= Q1 - 1.5 * IQR) & (all_data <= Q3 + 1.5 * IQR)
all_data = all_data[np.all(outlier_mask, axis=1)]

# Separar nuevamente X y y después de eliminar outliers
X_normalized = all_data[:, :-1]
y_normalized = all_data[:, -1]

# Dividir los datos normalizados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.3, random_state=42)

def select_columns(arr, columns):
    selected_columns = arr[:, np.where(columns)[0]]
    if selected_columns.shape[1] < 5:
        # Si la cantidad de columnas seleccionadas es menor que 5, rellenar con ceros
        selected_columns = np.pad(selected_columns, ((0, 0), (0, 5 - selected_columns.shape[1])), mode='constant')
    return selected_columns

# Lista para almacenar todas las soluciones y sus pérdidas
all_solutions = []

def objective_function(x):
    selected_columns = select_columns(X_test, x)
    loss = model.evaluate(selected_columns, y_test, verbose=0)
    all_solutions.append((x, loss))
    return loss  # Usamos el negativo para maximizar

# Definición del rango para cada dimensión (en este caso, 5 dimensiones)
lower_bound = [0, 0, 0, 0, 0]  # límite inferior para cada dimensión
upper_bound = [1, 1, 1, 1, 1]  # límite superior para cada dimensión

def clip_to_binary(x):
    return np.round(x).astype(int)

def constrained_objective_function(x):
    clipped_x = clip_to_binary(x)
    return objective_function(clipped_x)

# Definir la forma de entrada fija
input_shape = X_train.shape[1]

# Definir el modelo de red neuronal con capa de entrada fija
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=2)


# Ejecución de PSO
best_solution, best_value = pso(constrained_objective_function, lower_bound, upper_bound, maxiter=30, swarmsize=20)


sorted_solutions = sorted(all_solutions, key=lambda x: x[1])

printed_solutions = set()  # Conjunto para almacenar soluciones ya impresas
printed_count = 0  # Contador para llevar el registro de las soluciones impresas

unique_solutions = []  # Lista para almacenar las tres soluciones únicas

for solution, _ in sorted_solutions:
    rounded_solution = np.round(solution).astype(int)
    if rounded_solution.tolist() not in unique_solutions:
        unique_solutions.append(rounded_solution.tolist())
        print(f"Solución {len(unique_solutions)}: {rounded_solution}")
        if len(unique_solutions) == 3:
            break

print("Todas las soluciones únicas:", unique_solutions)

def build_model(solution):
    selected_columns = columns(solution, X_train)
    input_shape = selected_columns.shape[1]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def columns(indices_columnas, datos):
    columnas_seleccionadas = []

    # Iterar sobre los índices de las columnas y los valores del arreglo
    for indice, seleccionado in enumerate(indices_columnas):
        # Si el valor es 1, seleccionar la columna correspondiente
        if seleccionado == 1:
            # Agregar la columna al array de columnas seleccionadas
            columnas_seleccionadas.append(datos[:, indice]) 

    # Convertir la lista de columnas seleccionadas a un ndarray de NumPy
    columnas_seleccionadas = np.array(columnas_seleccionadas).T

    return columnas_seleccionadas


def evaluate_model(model, X_train, y_train, X_val, y_val, solution):
    train = columns(solution, X_train)
    test = columns(solution, X_val)
    history = model.fit(train, y_train, epochs=50, batch_size=32, validation_data=(test, y_val), verbose=2)
    val_loss = history.history['val_loss'][-1]
    return val_loss

best_val_loss = float('inf')
best_model = None
best_solution_vector = None

for solution in unique_solutions:
    model = build_model(solution)
    val_loss = evaluate_model(model, X_train, y_train, X_test, y_test, solution)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_solution_vector = solution

print("La mejor pérdida de validación encontrada:", best_val_loss)
print("El vector de la mejor solución encontrada:", best_solution_vector)

# Predecir los valores con el mejor modelo
y_pred = best_model.predict(columns(best_solution_vector,X_test ))

# Graficar los resultados reales vs. los predichos
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Reales', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Valores Predichos', alpha=0.5)
plt.title('Comparación de Valores Reales y Predichos')
plt.xlabel('Índice de Caso de Test')
plt.ylabel('Valor de Salida')
plt.legend()
plt.show()