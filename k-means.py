import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Cargamos el conjunto de datos de Iris
iris = load_iris()
X = iris.data  # características de las flores

# Aplicamos el algoritmo K-Means para agrupar las flores en 3 categorías (número de clusters)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Obtenemos los centroides de los clusters
centroids = kmeans.cluster_centers_
print("Centroides de los clusters:")
print(centroids)

# Obtenemos las etiquetas de los clusters para cada muestra
labels = kmeans.labels_

# Visualizamos los clusters
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')

# Graficamos los centroides de los clusters
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='red', label='Centroides')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Anchura del sépalo (cm)')
plt.title('Agrupamiento de flores de Iris usando K-Means')
plt.legend()
plt.show()