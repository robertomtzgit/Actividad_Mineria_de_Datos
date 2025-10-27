import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def run_kmeans(df, x_columns, n_clusters=3):
    """
    Ejecuta clustering K-Means sobre el dataset.
    
    Args:
        df: DataFrame con los datos
        x_columns: Lista de nombres de columnas para clustering
        n_clusters: Número de clusters a formar (default: 3)
    
    Returns:
        dict: Diccionario con resultados, métricas y gráficos
    """
    # Validaciones
    if df is None or df.empty:
        raise ValueError("El DataFrame está vacío o es None")
    
    if not x_columns:
        raise ValueError("Debes especificar al menos una columna para clustering")
    
    if n_clusters < 2:
        raise ValueError("El número de clusters debe ser al menos 2")
    
    # Verificar que las columnas existen
    missing_cols = [col for col in x_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas en el dataset: {missing_cols}")
    
    # Preparar datos
    X = df[x_columns].copy()
    
    # Eliminar filas con valores nulos
    X = X.dropna()
    
    if len(X) < n_clusters:
        raise ValueError(f"No hay suficientes datos válidos. Se necesitan al menos {n_clusters} muestras.")
    
    # Verificar que todas las columnas son numéricas
    if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("Todas las columnas deben ser numéricas para K-Means")
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Crear y entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calcular silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    
    # Reducir dimensionalidad para visualización (si hay más de 2 dimensiones)
    if len(x_columns) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
    else:
        X_pca = X_scaled
        explained_variance = [1.0, 1.0] if len(x_columns) == 2 else [1.0]
    
    # Gráfico 1: Visualización de clusters
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Colores para los clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = X_pca[labels == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[i]], label=f'Cluster {i}', 
                   alpha=0.6, edgecolors='k', s=80)
    
    # Marcar centroides
    if len(x_columns) > 2:
        centroids_pca = pca.transform(kmeans.cluster_centers_)
    else:
        centroids_pca = kmeans.cluster_centers_
    
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='X', s=300, edgecolors='black', 
               linewidths=2, label='Centroides', zorder=10)
    
    if len(x_columns) > 2:
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%} varianza)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%} varianza)', fontsize=12)
        ax1.set_title('K-Means Clustering (Proyección PCA)', fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel(x_columns[0] if len(x_columns) > 0 else 'Dimensión 1', fontsize=12)
        ax1.set_ylabel(x_columns[1] if len(x_columns) > 1 else 'Dimensión 2', fontsize=12)
        ax1.set_title('K-Means Clustering', fontsize=14, fontweight='bold')
    
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Añadir silhouette score
    textstr = f'Silhouette Score: {silhouette_avg:.4f}\nClusters: {n_clusters}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    buf1 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    plt.close()
    
    # Gráfico 2: Distribución de clusters
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de barras
    unique, counts = np.unique(labels, return_counts=True)
    bars = ax2.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Número de Muestras', fontsize=12)
    ax2.set_title('Distribución de Muestras por Cluster', fontsize=14, fontweight='bold')
    ax2.set_xticks(unique)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gráfico de pastel
    ax3.pie(counts, labels=[f'Cluster {i}' for i in unique], 
            colors=colors, autopct='%1.1f%%', startangle=90,
            explode=[0.05] * len(unique), shadow=True)
    ax3.set_title('Proporción de Muestras por Cluster', fontsize=14, fontweight='bold')
    
    buf2 = BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    plt.close()
    
    # Retornar resultados
    return {
        'labels': labels.tolist(),
        'silhouette_score': silhouette_avg,
        'n_clusters': n_clusters,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'inertia': kmeans.inertia_,
        'plot_clusters': buf1,
        'plot_distribution': buf2,
        'model': kmeans,
        'scaler': scaler
    }
