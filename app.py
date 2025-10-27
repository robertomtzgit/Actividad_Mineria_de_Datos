import streamlit as st
import pandas as pd
import io
from models.linear_regression_model import run_linear_regression
from models.logistic_regression_model import run_logistic_regression
from models.kmeans_model import run_kmeans

# Configuración de la página
st.set_page_config(
    page_title="Proyecto de Minería de Datos",
    page_icon="📊",
    layout="wide"
)

# Inicializar session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

def reset_session():
    """Reinicia la sesión y limpia todos los datos"""
    st.session_state.df = None
    st.session_state.file_uploaded = False
    st.rerun()

def load_file(uploaded_file):
    """Carga un archivo CSV o XLSX y retorna un DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Formato no soportado. Por favor, sube un archivo CSV o XLSX.")
            return None
        
        if df.empty:
            st.error("❌ El archivo está vacío.")
            return None
        
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        return None

# Título principal
st.title("📊 Proyecto de Minería de Datos")
st.markdown("---")

# Sidebar para controles principales
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Botón de reinicio
    if st.button("🔄 Reiniciar Sesión", use_container_width=True):
        reset_session()
    
    st.markdown("---")
    
    # Carga de archivo
    st.subheader("📁 Cargar Datos")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV o XLSX",
        type=['csv', 'xlsx', 'xls'],
        help="Sube tu dataset para comenzar el análisis"
    )
    
    if uploaded_file is not None and not st.session_state.file_uploaded:
        df = load_file(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.rerun()

# Contenido principal
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Información del dataset
    st.subheader("📋 Información del Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", df.shape[0])
    with col2:
        st.metric("Columnas", df.shape[1])
    with col3:
        st.metric("Columnas Numéricas", len(df.select_dtypes(include=['number']).columns))
    
    # Vista previa de datos
    st.subheader("👀 Vista Previa de Datos (primeras 100 filas)")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Información de columnas
    with st.expander("ℹ️ Información de Columnas"):
        col_info = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values,
            'Valores Nulos': df.isnull().sum().values,
            'Valores Únicos': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    st.markdown("---")
    
    # Selección de algoritmo
    st.subheader("🤖 Selección de Algoritmo")
    algorithm = st.selectbox(
        "Elige el algoritmo a ejecutar:",
        ["Selecciona un algoritmo...", "Regresión Lineal Múltiple", "Regresión Logística Binaria", "K-Means"]
    )
    
    if algorithm != "Selecciona un algoritmo...":
        st.markdown("---")
        
        # REGRESIÓN LINEAL MÚLTIPLE
        if algorithm == "Regresión Lineal Múltiple":
            st.subheader("📈 Regresión Lineal Múltiple")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Se necesitan al menos 2 columnas numéricas para regresión lineal.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Variables Independientes (X)**")
                    x_cols = st.multiselect(
                        "Selecciona las columnas para X:",
                        numeric_cols,
                        help="Puedes seleccionar múltiples columnas"
                    )
                
                with col2:
                    st.markdown("**Variable Dependiente (y)**")
                    available_y = [col for col in numeric_cols if col not in x_cols]
                    y_col = st.selectbox(
                        "Selecciona la columna para y:",
                        [""] + available_y
                    )
                
                if st.button("▶️ Ejecutar Regresión Lineal", type="primary", use_container_width=True):
                    if not x_cols or not y_col:
                        st.error("❌ Debes seleccionar al menos una variable independiente y una dependiente.")
                    else:
                        with st.spinner("Ejecutando regresión lineal..."):
                            try:
                                results = run_linear_regression(df, x_cols, y_col)
                                
                                st.success("✅ Regresión lineal completada")
                                
                                # Mostrar resultados
                                st.subheader("📊 Resultados")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R² Score", f"{results['r2']:.4f}")
                                    st.metric("RMSE", f"{results['rmse']:.4f}")
                                
                                with col2:
                                    st.markdown("**Coeficientes:**")
                                    coef_df = pd.DataFrame({
                                        'Variable': x_cols,
                                        'Coeficiente': results['coefficients']
                                    })
                                    st.dataframe(coef_df, use_container_width=True)
                                    st.metric("Intercepto", f"{results['intercept']:.4f}")
                                
                                # Mostrar gráficos
                                st.subheader("📉 Visualizaciones")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.image(results['plot_predictions'], caption="Valores Reales vs Predichos")
                                
                                with col2:
                                    st.image(results['plot_residuals'], caption="Distribución de Residuos")
                                
                            except Exception as e:
                                st.error(f"❌ Error al ejecutar regresión lineal: {str(e)}")
        
        # REGRESIÓN LOGÍSTICA BINARIA
        elif algorithm == "Regresión Logística Binaria":
            st.subheader("🎯 Regresión Logística Binaria")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            all_cols = df.columns.tolist()
            
            if len(numeric_cols) < 1:
                st.error("❌ Se necesita al menos 1 columna numérica para las variables independientes.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Variables Independientes (X)**")
                    x_cols = st.multiselect(
                        "Selecciona las columnas para X:",
                        numeric_cols,
                        help="Selecciona las características para el modelo"
                    )
                
                with col2:
                    st.markdown("**Variable Dependiente (y) - Binaria**")
                    y_col = st.selectbox(
                        "Selecciona la columna para y:",
                        [""] + all_cols
                    )
                    
                    if y_col:
                        unique_vals = df[y_col].nunique()
                        st.info(f"ℹ️ La columna '{y_col}' tiene {unique_vals} valores únicos")
                        if unique_vals != 2:
                            st.warning("⚠️ La regresión logística binaria requiere exactamente 2 clases.")
                
                if st.button("▶️ Ejecutar Regresión Logística", type="primary", use_container_width=True):
                    if not x_cols or not y_col:
                        st.error("❌ Debes seleccionar al menos una variable independiente y una dependiente.")
                    else:
                        with st.spinner("Ejecutando regresión logística..."):
                            try:
                                results = run_logistic_regression(df, x_cols, y_col)
                                
                                st.success("✅ Regresión logística completada")
                                
                                # Mostrar métricas
                                st.subheader("📊 Métricas de Clasificación")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                                with col2:
                                    st.metric("Precision", f"{results['precision']:.4f}")
                                with col3:
                                    st.metric("Recall", f"{results['recall']:.4f}")
                                with col4:
                                    st.metric("F1-Score", f"{results['f1']:.4f}")
                                
                                # Matriz de confusión
                                st.subheader("🔢 Matriz de Confusión")
                                st.dataframe(results['confusion_matrix_df'], use_container_width=True)
                                
                                # Mostrar gráficos
                                st.subheader("📉 Visualizaciones")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.image(results['plot_confusion_matrix'], caption="Matriz de Confusión")
                                
                                with col2:
                                    st.image(results['plot_roc_curve'], caption="Curva ROC")
                                
                            except Exception as e:
                                st.error(f"❌ Error al ejecutar regresión logística: {str(e)}")
        
        # K-MEANS
        elif algorithm == "K-Means":
            st.subheader("🎨 Clustering K-Means")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Se necesitan al menos 2 columnas numéricas para K-Means.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Variables para Clustering**")
                    x_cols = st.multiselect(
                        "Selecciona las columnas:",
                        numeric_cols,
                        help="Selecciona las características para el clustering"
                    )
                
                with col2:
                    st.markdown("**Número de Clusters**")
                    n_clusters = st.slider(
                        "Selecciona el número de clusters (k):",
                        min_value=2,
                        max_value=10,
                        value=3,
                        help="Número de grupos a formar"
                    )
                
                if st.button("▶️ Ejecutar K-Means", type="primary", use_container_width=True):
                    if not x_cols:
                        st.error("❌ Debes seleccionar al menos una variable.")
                    else:
                        with st.spinner("Ejecutando K-Means..."):
                            try:
                                results = run_kmeans(df, x_cols, n_clusters)
                                
                                st.success("✅ K-Means completado")
                                
                                # Mostrar métricas
                                st.subheader("📊 Resultados del Clustering")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
                                    st.info("ℹ️ Valores cercanos a 1 indican clusters bien definidos")
                                
                                with col2:
                                    st.metric("Número de Clusters", n_clusters)
                                    st.metric("Muestras Totales", len(results['labels']))
                                
                                # Distribución de clusters
                                st.subheader("📈 Distribución de Clusters")
                                cluster_counts = pd.Series(results['labels']).value_counts().sort_index()
                                cluster_df = pd.DataFrame({
                                    'Cluster': cluster_counts.index,
                                    'Cantidad': cluster_counts.values,
                                    'Porcentaje': (cluster_counts.values / len(results['labels']) * 100).round(2)
                                })
                                st.dataframe(cluster_df, use_container_width=True)
                                
                                # Mostrar gráficos
                                st.subheader("📉 Visualizaciones")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.image(results['plot_clusters'], caption="Visualización de Clusters")
                                
                                with col2:
                                    st.image(results['plot_distribution'], caption="Distribución de Clusters")
                                
                                # Descargar resultados
                                st.subheader("💾 Descargar Resultados")
                                result_df = df[x_cols].copy()
                                result_df['Cluster'] = results['labels']
                                
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar CSV con Clusters",
                                    data=csv,
                                    file_name="kmeans_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error al ejecutar K-Means: {str(e)}")

else:
    # Pantalla de bienvenida
    st.info("👈 Por favor, carga un archivo CSV o XLSX desde la barra lateral para comenzar.")
    
    st.markdown("""
    ### 📚 Instrucciones de Uso
    
    1. **Carga tu dataset**: Usa el selector de archivos en la barra lateral
    2. **Explora tus datos**: Revisa la vista previa y la información de columnas
    3. **Selecciona un algoritmo**: Elige entre Regresión Lineal, Regresión Logística o K-Means
    4. **Configura las variables**: Selecciona las columnas apropiadas para tu análisis
    5. **Ejecuta el modelo**: Haz clic en el botón de ejecución y revisa los resultados
    
    ### 🎯 Algoritmos Disponibles
    
    - **Regresión Lineal Múltiple**: Predice valores continuos basándose en múltiples variables
    - **Regresión Logística Binaria**: Clasifica datos en dos categorías
    - **K-Means**: Agrupa datos similares en clusters
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Proyecto de Minería de Datos | Desarrollado con Streamlit 🚀</div>",
    unsafe_allow_html=True
)
