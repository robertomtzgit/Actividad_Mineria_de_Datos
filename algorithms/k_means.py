import pandas as pd
from google.colab import files
import io
import numpy as np

# Información requerida a pedir al usuario

# Archivo
print('Usuario, ingrese el archivo a poder ejecutar el programa')
uploaded = files.upload()
# Número de iteraciones
iteraciones = int(input('Usuario, digite la cantidad de iteraciones máximas que requiere para ejecutar el programa: '))
# Número de clústers
k = int(input('Usuario, digite la cantidad de clústers que requiere para ejecutar el programa: '))

# Obtener headers y filas

archivo = next(iter(uploaded))  # primer archivo
df = pd.read_csv(io.BytesIO(uploaded[archivo]))
ids = df['ID']
df_num = df.drop(columns=['ID']).select_dtypes(include=['number']) # Obtenemos toda la info con la que se hace el procesado, números pero sin contar el id
headers = df.columns

print(df_num)

# Funciones auxiliares

def obtenerDistanciaEuclidiana(vector1, vector2):
    # print(vector1, vector2)
    distancia = 0
    for i in range(len(vector1)):
        distancia += (vector1[i] - vector2[i])**2
    return distancia**(1/2)

def definirCentroidesRandom(df, k):
    centroide = []
    for _ in range(k):  # para cada centroide
        punto = []
        for col in df.columns:  # por columna
            min_val = df[col].min()
            max_val = df[col].max()
            random_val = np.random.uniform(min_val, max_val)
            punto.append(random_val)
        centroide.append(punto)
    return np.array(centroide)

def centroidesRecalculados(df, labels, k):
    df_array = df.to_numpy()
    labels_array = np.array(labels)
    n_features = df_array.shape[1]
    nuevos_centroides = np.zeros((k, n_features))

    for i in range(k):
        filas_cluster = df_array[labels_array == i]
        if len(filas_cluster) > 0:
            nuevos_centroides[i] = np.mean(filas_cluster, axis=0)
    return nuevos_centroides

def asignarClusters(df, centroides):
    labels = []
    data = df.to_numpy()
    for fila in data:
        distancias = [obtenerDistanciaEuclidiana(fila, c) for c in centroides]
        labels.append(np.argmin(distancias))
    return labels

def calculoError(df, labels, centroides, k):
    errores = []
    data = df.to_numpy()
    labels_array = np.array(labels)

    for i in range(k):
        filas_cluster = data[labels_array == i]
        if len(filas_cluster) > 0:
            error_cluster = np.sum((filas_cluster - centroides[i])**2)
        else:
            error_cluster = 0
        errores.append(error_cluster)

    error_total = sum(errores)

    return errores, error_total

# Procesamiento de kmeans
centroides = None
centroides_anteriores = None
for iteracion in range(iteraciones):
    print(f'Iteración: {int(iteracion) + 1}')

    if centroides is None or len(centroides) == 0:
        centroides = definirCentroidesRandom(df_num, k)

    labels = asignarClusters(df_num, centroides)

    errores, error_total = calculoError(df_num, labels, centroides, k)

    for i, e in enumerate(errores):
        print(f'Error del cluster {i}: {e:.4f}')
    print(f'Error total del agrupamiento: {error_total:.4f}')

    centroides = centroidesRecalculados(df_num, labels, k)

    if centroides_anteriores is not None and np.array_equal(centroides, centroides_anteriores):
        print(f'Centroides iguales, existe un paro en la iteración: {int(iteracion) + 1}')
        break

    centroides_anteriores = centroides

df_resultado = df.copy()
df_resultado = df_resultado.set_index('ID')
df_resultado['cluster'] = labels
# Mostrar todas las filas
pd.set_option('display.max_rows', None)
# Mostrar todas las columnas
pd.set_option('display.max_columns', None)
# Ajusta ancho de pantalla
pd.set_option('display.width', 1000)
print(df_resultado)


