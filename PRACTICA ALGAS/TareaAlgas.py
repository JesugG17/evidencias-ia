import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

print("Is GPU available:", tf.config.list_physical_devices('GPU'))
# Ruta a la carpeta de imágenes
folder_path = os.path.join(os.path.dirname(__file__), "IMAGENES")

# Obtener la lista de archivos de imágenes en la carpeta
image_files = os.listdir(folder_path)

# Preparar los datos
file_name = "C:\\Users\\alanm\\Desktop\\Inteligencia artificial\\Practicas\\Ultima unidad\\PRACTICA ALGAS\\Datos frames.xlsx"
excel_pathP = os.path.join(os.path.dirname(__file__), file_name)

# Leer el archivo Excel
data_frame = pd.read_excel(excel_path)
labels = data_frame["Nombre imagen"]
biomass = data_frame["Biomasa g/L"]

# Leer todas las imágenes y convertirlas en matrices
images = []
for label in labels:
    # Verificar si el archivo de imagen existe
    if f"{label}.png" in image_files:
        # Construir la ruta completa a la imagen
        image_path = os.path.join(folder_path, f"{label}.png")
        # Cargar la imagen y convertirla en una matriz
        img = load_img(image_path, target_size=(124, 124)) 
        img_array = img_to_array(img)
        images.append(img_array)
    else:
        print(f"La imagen {label}.png no fue encontrada.")

images = np.array(images)

# Función para normalizar las imágenes
def normalize_images(images):
    return images / 255.0

# Normalizar las imágenes
images = normalize_images(images)

#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, biomass, test_size=0.3, random_state=42)

# Construir el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

#Compilar el modelo y entrenarlo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=350, batch_size=32, validation_data=(X_test, y_test))

#Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print("Error en el conjunto de prueba:", loss)

#Realizar predicciones
predictions = model.predict(X_test)

#Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, c='blue', label='Biomasa Predicha')
plt.scatter(y_test, y_test, c='red', label='Biomasa Real')
plt.xlabel("Biomasa Real")
plt.ylabel("Biomasa Predicha")
plt.title("Predicciones vs. Realidad")
plt.legend()
plt.show()
