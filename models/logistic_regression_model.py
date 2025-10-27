import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def run_logistic_regression(df, x_columns, y_column):
    """
    Ejecuta regresión logística binaria sobre el dataset.
    
    Args:
        df: DataFrame con los datos
        x_columns: Lista de nombres de columnas para variables independientes
        y_column: Nombre de la columna para variable dependiente (binaria)
    
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
    
    # Verificar que X es numérico
    if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("Todas las columnas de X deben ser numéricas")
    
    # Codificar y si no es numérico
    le = None
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Verificar que y es binario
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError(f"La variable dependiente debe ser binaria. Se encontraron {len(unique_classes)} clases únicas.")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crear y entrenar el modelo
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Crear DataFrame de matriz de confusión
    class_names = le.classes_ if le else unique_classes
    cm_df = pd.DataFrame(
        cm,
        index=[f'Real {class_names[0]}', f'Real {class_names[1]}'],
        columns=[f'Pred {class_names[0]}', f'Pred {class_names[1]}']
    )
    
    # Gráfico 1: Matriz de confusión
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, linewidths=1, linecolor='black',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax1.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    
    # Añadir métricas
    textstr = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(1.5, 0.5, textstr, transform=ax1.transData, fontsize=10,
                verticalalignment='center', bbox=props)
    
    buf1 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    plt.close()
    
    # Gráfico 2: Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Clasificador Aleatorio')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
    ax2.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    ax2.set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    buf2 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    plt.close()
    
    # Retornar resultados
    return {
        'confusion_matrix': cm,
        'confusion_matrix_df': cm_df,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'plot_confusion_matrix': buf1,
        'plot_roc_curve': buf2,
        'model': model,
        'label_encoder': le
    }
