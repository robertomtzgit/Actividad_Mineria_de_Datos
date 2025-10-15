import pandas as pd
import numpy as np

# Posibles datasets
dataset_1 = 'logistic_dataset_educacion.csv'
dataset_2 = 'logistic_dataset_negocios.csv'
dataset_3 = 'logistic_dataset_salud.csv'

dataset_usar = dataset_2

# Pedir al usaurio los parametros requeridos
learning_rate = float(input('Ingrese el learning rate a utilizar: '))
iteraciones = int(input('Ingrese el número de iteraciones a utilizar: '))

# Gestionar el balanceo de datos
def checkBalanceo(df, umbral=0.02, porcentaje_objetivo=0.3):
    clases, conteos = np.unique(df.iloc[:, -1], return_counts=True)
    total = sum(conteos)
    distribucion = dict(zip(clases, conteos))
    #print(distribucion)

    for clase, conteo in distribucion.items():
        proporcion = conteo / total
        if proporcion < umbral:
            filas_minoritarias = df[df.iloc[:, -1] == clase]
            objetivo = int(porcentaje_objetivo * total)
            repeticiones = (objetivo // conteo) + 1
            duplicados = pd.concat([filas_minoritarias] * repeticiones, ignore_index=True)
            df = pd.concat([df, duplicados], ignore_index=True)
            print('Balanceo aplicado')
            break
    else:
        print("No se detectó necesidad de balancear")
    return df

# Funciones de calculo

def sigmoide(z):
    z = np.clip(z, -500, 500)  # Evita valores extremos
    return 1 / (1 + np.exp(-z))

def calc_z(tetha, xi):
    return np.dot(tetha, xi)

def calc_h(tetha, xi):
    z = calc_z(tetha, xi)
    #print(f"Valor de z: {z}")
    return sigmoide(z)

def calc_tetha(tetha, xi, yi, alpha, h):
    return tetha - alpha * (h - yi) * xi

def calc_verosimilitud(tetha, X, y):
    h = sigmoide(np.dot(X, tetha))
    h = np.clip(h, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

datos = pd.read_csv(dataset_usar)

# en dado caso de que tengamos extremos (tipo 119 - 1) hago un balanceo
datos = checkBalanceo(datos)

independientes = datos.iloc[:, :-1].astype(float).values # Ultima
dependientes = datos.iloc[:, -1].astype(float).values # Los demás

# Normalización
media = np.mean(independientes, axis=0)
desviacion = np.std(independientes, axis=0)
independientes = (independientes - media) / desviacion

independientes = np.hstack([np.ones((independientes.shape[0], 1)), independientes]) # añadir el 1
tetha = np.zeros(independientes.shape[1], dtype=float)

# Gradiente y muestra del valor de verosimilitud
for i in range(iteraciones):
    for xi, yi in zip(independientes, dependientes):
        h = calc_h(tetha, xi)
        #print(f"Valor de h: {h}")
        tetha = calc_tetha(tetha, xi, yi, learning_rate, h)
        #print(f"Valor de tetha: {tetha}")
    verosimilitud = calc_verosimilitud(tetha, independientes, dependientes)
    print(f"Iteración {i+1}: Verosimilitud = {verosimilitud:.4f}")
    if i > 0 and abs(verosimilitud_anterior - verosimilitud) < 1e-6: # converge si la diferencia es solamente de una millonesima
        print("Convergencia alcanzada.")
        break
    verosimilitud_anterior = verosimilitud


print("Tetha final:", tetha)

# Predicciones
z = np.dot(independientes, tetha)
probabilidades = sigmoide(z)
predicciones_binarias = (probabilidades >= 0.5).astype(int)

datos['Probabilidad'] = probabilidades
datos['Prediccion'] = predicciones_binarias

# dataframe sin cortarrrrr
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(datos)

# Funciones para el modelo
def matriz_confusion(y_true, y_pred):
    vp = np.sum((y_true == 1) & (y_pred == 1))
    vn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return vp, vn, fp, fn

def exactitud(vp, vn, fp, fn):
    return (vp + vn) / (vp + vn + fp + fn)

def precision(vp, fp):
    return vp / (vp + fp) if (vp + fp) != 0 else 0

def recall(vp, fn):
    return vp / (vp + fn) if (vp + fn) != 0 else 0

# Output final formateado
y_true = dependientes
y_pred = (probabilidades >= 0.5).astype(int)

# Calcular métricas
vp, vn, fp, fn = matriz_confusion(y_true, y_pred)
acc = exactitud(vp, vn, fp, fn)
prec = precision(vp, fp)
rec = recall(vp, fn)

resultados = pd.DataFrame({
    'ID': np.arange(len(y_true)),
    'Probabilidad': probabilidades,
    'Prediccion': y_pred,
    'Real': y_true
})

resultados['Exactitud'] = acc
resultados['Precision'] = prec
resultados['Recall'] = rec

print(resultados)

