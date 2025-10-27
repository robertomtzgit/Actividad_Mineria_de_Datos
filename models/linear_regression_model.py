import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def run_linear_regression(df, x_columns, y_column):
    """
    Ejecuta regresión lineal múltiple sobre el dataset.
    
    Args:
        df: DataFrame con los datos
        x_columns: Lista de nombres de columnas para variables independientes
        y_column: Nombre de la columna para variable dependiente
    
    Returns:
        dict: Diccionario con resultados, métricas y gráficos
    """
    # Validaciones
    if df is None or df.empty:
        raise ValueError("El DataFrame está vacío o es None")
    
    if not x_columns or not y_column:
        raise ValueError("Debes especificar columnas para X e y")
    
    # Verificar que las columnas existen
    missing_cols = [col for col in x_columns + [y_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing_cols}")
    
    # Preparar datos
    X = df[x_columns].copy()
    y = df[y_column].copy()
    
    # Eliminar filas con valores nulos
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 2:
        raise ValueError("No hay suficientes datos válidos después de eliminar valores nulos")
    
    # Verificar que todas las columnas son numéricas
    if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("Todas las columnas de X deben ser numéricas")
    
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("La columna y debe ser numérica")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calcular métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Usar métricas de test para el reporte principal
    r2 = r2_test
    rmse = rmse_test
    
    # Coeficientes
    coefficients = model.coef_.tolist()
    intercept = model.intercept_
    
    # Crear gráficos
    # Gráfico 1: Valores reales vs predichos
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=80)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Predicción perfecta')
    ax1.set_xlabel('Valores Reales', fontsize=12)
    ax1.set_ylabel('Valores Predichos', fontsize=12)
    ax1.set_title('Regresión Lineal: Valores Reales vs Predichos', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir métricas al gráfico
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    buf1 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    plt.close()
    
    # Gráfico 2: Distribución de residuos
    residuals = y_test - y_pred_test
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de residuos
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('Residuos', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Residuo = 0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico de residuos vs predichos
    ax3.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k', s=80)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Valores Predichos', fontsize=12)
    ax3.set_ylabel('Residuos', fontsize=12)
    ax3.set_title('Residuos vs Valores Predichos', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    buf2 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    plt.close()
    
    # Retornar resultados
    return {
        'coefficients': coefficients,
        'intercept': intercept,
        'r2': r2,
        'rmse': rmse,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'plot_predictions': buf1,
        'plot_residuals': buf2,
        'model': model
    }
