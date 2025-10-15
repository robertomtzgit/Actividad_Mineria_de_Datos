import numpy as np
import pandas as pd
from math import sqrt

pd.set_option("display.precision", 4)

# Configuracion de datasets

datasets_info = {
    1: {
        "file": "multivar_dataset_calificacion.csv",
        "X_cols": ["horas_estudio", "horas_sueno", "horas_redes"],
        "y_col": "calificacion"
    },
    2: {
        "file": "multivar_dataset_presionArterial.csv",
        "X_cols": ["edad", "bmi", "sodio_g_dia"],
        "y_col": "sbp_mmHg"
    },
    3: {
        "file": "multivar_dataset_ventas.csv",
        "X_cols": ["ads_mxn", "descuento_pct", "temperatura_c"],
        "y_col": "ventas"
    }
}

print("=== SELECCIÓN DE DATASET ===")
print("1) multivar_dataset_calificacion.csv")
print("2) multivar_dataset_presionArterial.csv")
print("3) multivar_dataset_ventas.csv")
opcion = int(input("Seleccione el dataset a usar (1-3): "))
info = datasets_info.get(opcion, datasets_info[2])  # default dataset 2

df = pd.read_csv(info["file"])
print("\nDataset cargado:\n")
print(df.head())

# Funciones auxiliares

def add_bias(X):
    """Agrega columna de 1's para el bias (θ0)."""
    return np.c_[np.ones(X.shape[0]), X]

def compute_metrics(y_true, y_pred):
    """Calcula R2, MSE y RMSE."""
    mse = np.mean((y_true - y_pred)**2)
    rmse = sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float('nan')
    return r2, mse, rmse

def sgd_linear_regression(X, y, alpha=0.01, epochs=10, verbose=True):
    """Entrena regresión lineal múltiple con SGD."""
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    history = []

    for epoch in range(epochs):
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        for i in idx:
            xi = X[i]
            yi = y[i]
            pred = np.dot(theta, xi)
            error = pred - yi
            grad = error * xi
            theta -= alpha * grad

        preds = X.dot(theta)
        r2, mse, rmse = compute_metrics(y, preds)
        history.append((epoch+1, mse, rmse, r2))
        if verbose and (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} -> MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    return theta, history

# Preparacion de variables

# Normalizacion de X
X_raw = df[info["X_cols"]].values
y = df[info["y_col"]].values

X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_norm = (X_raw - X_mean) / X_std
X = add_bias(X_norm)

# Demo manual de 2 primeras iteraciones
alpha_demo = 0.01
theta_demo = np.zeros(X.shape[1])
print("\n=== DEMOSTRACIÓN MANUAL DE LAS 2 PRIMERAS ACTUALIZACIONES ===")
for step in range(2):
    xi = X[step]
    yi = y[step]
    pred = np.dot(theta_demo, xi)
    error = pred - yi
    grad = error * xi
    theta_demo -= alpha_demo * grad
    print(f"\nIteración {step+1}")
    print(f"x = {xi}, y = {yi}")
    print(f"Predicción = {pred:.4f}, Error = {error:.4f}")
    print(f"Nuevos θ = {theta_demo}")

# Entrenamiento completo

print("\n=== ENTRENAMIENTO DEL MODELO ===")
alpha = float(input("Ingrese el valor de alpha (learning rate): "))
epochs = int(input("Ingrese el número de iteraciones (m): "))

theta, history = sgd_linear_regression(X, y, alpha=alpha, epochs=epochs, verbose=True)

print("\nParámetros θ encontrados:")
for i, val in enumerate(theta):
    print(f"θ{i} = {val:.4f}")

# Modelo resultante
equation = f"{info['y_col']} = {theta[0]:.4f}"
for i, col in enumerate(info["X_cols"]):
    equation += f" + {theta[i+1]:.4f}*{col}_norm"
print("\nModelo resultante (variables normalizadas):")
print(equation)

# Metricas finales
y_pred = X.dot(theta)
r2, mse, rmse = compute_metrics(y, y_pred)
print("\nMétricas finales en todo el dataset:")
print(f"R2 = {r2:.4f} | MSE = {mse:.4f} | RMSE = {rmse:.4f}")

# Evaluacion en 5 instancias

indices = np.random.choice(len(X), size=5, replace=False)
rows = []
for idx in indices:
    y_est = X[idx].dot(theta)
    y_true = y[idx]
    rows.append({
        "ID_X": int(idx),
        "Y_estimada": float(y_est),
        "Y_esperada": float(y_true)
    })

df5 = pd.DataFrame(rows)

# Métricas globales sobre las 5 instancias
r2_5, mse_5, rmse_5 = compute_metrics(df5["Y_esperada"].values, df5["Y_estimada"].values)

print("\n=== Evaluación sobre 5 instancias ===")
print(df5.to_string(index=False))
print("\nMétricas sobre estas 5 instancias:")
print(f"R2 = {r2_5:.4f} | MSE = {mse_5:.4f} | RMSE = {rmse_5:.4f}")
