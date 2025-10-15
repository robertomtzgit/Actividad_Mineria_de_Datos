import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# -------------------------- Helpers --------------------------

def load_dataframe(uploaded_file):
    """Carga un archivo CSV o XLSX y devuelve un DataFrame."""
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Formato no soportado. Usa .csv o .xlsx")
            return None
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None


def validate_dataframe(df):
    """Valida que el dataframe tenga al menos 1 fila y 1 columna numérica."""
    if df is None:
        return False, "No hay dataframe cargado"
    if df.shape[0] < 1:
        return False, "El archivo debe contener al menos una fila de datos"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        return False, "El archivo debe contener al menos una columna numérica"
    return True, "OK"


# -------------------------- Model runners --------------------------

def run_linear_regression(df, features, target):
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coef = model.coef_
    intercept = model.intercept_
    r2 = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    results = {
        'coeficientes': coef.tolist(),
        'intercepto': float(intercept),
        'r2': float(r2),
        'rmse': float(rmse)
    }
    return results


def run_logistic_regression(df, features, target):
    # Prepara X e y; si y no es binaria, se intenta codificar
    X = df[features].values
    y_raw = df[target]
    le = LabelEncoder()
    try:
        y = le.fit_transform(y_raw)
    except Exception:
        # intentar convertir a 0/1 si son numéricos
        y = y_raw.values

    # Validar binariedad
    unique_vals = np.unique(y)
    if unique_vals.shape[0] != 2:
        raise ValueError("La variable objetivo debe ser binaria para regresión logística")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    results = {
        'matriz_confusion': cm.tolist(),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    return results


def run_kmeans(df, features, k):
    X = df[features].values
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    labels = model.labels_
    inertia = float(model.inertia_)
    # contar miembros por cluster
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(map(int, unique.tolist()), counts.tolist()))

    results = {
        'labels': labels.tolist(),
        'inertia': inertia,
        'sizes': sizes
    }
    return results


# -------------------------- App layout --------------------------

st.set_page_config(page_title="Proyecto Miner\u00eda de Datos", layout='wide')
st.title("Proyecto: Aplicación de Minería de Datos")
st.markdown("Cargar un dataset (CSV/XLSX), seleccionar variables y ejecutar ML (scikit-learn).")

# Sidebar: controles principales
with st.sidebar:
    st.header("Controles")
    uploaded_file = st.file_uploader("Cargar archivo CSV o XLSX", type=['csv', 'xls', 'xlsx'])
    st.markdown("---")
    btn_reset = st.button("Reiniciar sesión / Limpiar")

# Reinicio simple usando session state
if btn_reset:
    for key in list(st.session_state.keys()):
        try:
            del st.session_state[key]
        except Exception:
            pass
    st.experimental_rerun()

# Main area
if uploaded_file is not None:
    df = load_dataframe(uploaded_file)
    valid, message = validate_dataframe(df)
    if not valid:
        st.error(message)
    else:
        st.success("Archivo cargado y validado ✅")
        st.subheader("Vista previa (primeras 100 filas)")
        st.dataframe(df.head(100))

        # Columnas numéricas y todas las columnas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        st.sidebar.subheader("Selección de variables")
        features = st.sidebar.multiselect("Variables independientes (X) - elegir al menos 1", options=numeric_cols)
        target = None
        algo = st.sidebar.selectbox("Seleccionar algoritmo", options=['Regresión lineal múltiple', 'Regresión logística binaria', 'K-means'])

        if algo in ['Regresión lineal múltiple', 'Regresión logística binaria']:
            # Para regresiones damos la opción de elegir cualquier columna (preferible numérica para lineal)
            target = st.sidebar.selectbox("Variable dependiente (Y)", options=all_cols)

        if algo == 'K-means':
            k = st.sidebar.number_input("Número de clusters (k)", min_value=1, max_value=20, value=3, step=1)

        run_btn = st.sidebar.button("Ejecutar algoritmo")

        # Ejecutar cuando el usuario presione
        if run_btn:
            try:
                if algo == 'Regresión lineal múltiple':
                    if not features or target is None:
                        st.error("Selecciona al menos 1 variable independiente y la variable dependiente.")
                    else:
                        # Asegurarse de que target y features sean numéricos
                        if target not in numeric_cols:
                            st.warning("Advertencia: la variable dependiente no es numérica. Se intentará convertir.")
                        results = run_linear_regression(df, features, target)
                        st.subheader("Resultados - Regresión Lineal Múltiple")
                        st.write("Coeficientes:")
                        coef_df = pd.DataFrame({'feature': features, 'coefficient': results['coeficientes']})
                        st.table(coef_df)
                        st.write(f"Intercepto: {results['intercepto']}")
                        st.write(f"R²: {results['r2']:.4f}")
                        st.write(f"RMSE: {results['rmse']:.4f}")

                elif algo == 'Regresión logística binaria':
                    if not features or target is None:
                        st.error("Selecciona variables e la variable dependiente.")
                    else:
                        # Intentar ejecutar y capturar si no es binaria
                        try:
                            results = run_logistic_regression(df, features, target)
                            st.subheader("Resultados - Regresión Logística Binaria")
                            st.write("Matriz de confusión:")
                            st.table(results['matriz_confusion'])
                            st.write(f"Accuracy: {results['accuracy']:.4f}")
                            st.write(f"Precision: {results['precision']:.4f}")
                            st.write(f"Recall: {results['recall']:.4f}")
                            st.write(f"F1-score: {results['f1_score']:.4f}")
                        except ValueError as ve:
                            st.error(str(ve))

                elif algo == 'K-means':
                    if not features:
                        st.error("Selecciona al menos 1 variable numérica para K-means.")
                    else:
                        results = run_kmeans(df, features, k)
                        st.subheader("Resultados - K-means")
                        st.write(f"Inercia: {results['inertia']}")
                        st.write("Tamaños de grupos:")
                        sizes_df = pd.DataFrame(list(results['sizes'].items()), columns=['cluster', 'size'])
                        st.table(sizes_df)
                        # mostrar etiquetas como una nueva columna (opcional)
                        if st.checkbox("Agregar etiquetas al dataframe (columna: kmeans_label)"):
                            df_with_labels = df.copy()
                            df_with_labels['kmeans_label'] = results['labels']
                            st.dataframe(df_with_labels.head(100))

            except Exception as e:
                st.error(f"Error ejecutando el algoritmo: {e}")

else:
    st.info("Carga un archivo en la barra lateral para comenzar")

# -------------------------- Footer / notas --------------------------
st.markdown("---")
st.write("Notas:")
st.write("- Esta versión inicial permite cargar y validar datasets, seleccionar variables y ejecutar los tres algoritmos pedidos.")
st.write("- Próximos pasos sugeridos: añadir validaciones más robustas, manejo de valores faltantes, y mejores mensajes de error.")

