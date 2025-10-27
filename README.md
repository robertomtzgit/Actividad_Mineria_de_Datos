# 📊 Proyecto de Minería de Datos

Aplicación web interactiva para análisis de datos y machine learning desarrollada con Streamlit.

## 🚀 Características

- **Regresión Lineal Múltiple**: Predice valores continuos basándose en múltiples variables
- **Regresión Logística Binaria**: Clasifica datos en dos categorías
- **K-Means Clustering**: Agrupa datos similares en clusters

## 📋 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🔧 Instalación

1. Clona o descarga este repositorio

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## ▶️ Ejecución

Para iniciar la aplicación, ejecuta:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🌐 Deploy

El proyecto está desplegado online en Streamlit Community Cloud y puedes probarlo aquí:

[Acceder a la App de la actividad](https://actividaddemineriadedatoscucei.streamlit.app/)

## 📁 Estructura del Proyecto

```
proyecto_mineria/
│
├── app.py                              # Aplicación principal de Streamlit
├── requirements.txt                    # Dependencias del proyecto
├── README.md                           # Este archivo
│
├── models/                            # Módulos de algoritmos
│   ├── linear_regression_model.py     # Regresión lineal múltiple
│   ├── logistic_regression_model.py   # Regresión logística binaria
│   └── kmeans_model.py                # Clustering K-Means
│
├── algorithms/                        # Algoritmos realizados por nosotros
│   ├── multiple_linear_regression.py  
│   ├── logistic_regression.py         
│   └── k_means.py 
|
└── data/                              # Datasets de ejemplo
    ├── multivar_dataset_calificacion.csv
    ├── logistic_dataset_educacion.csv
    └── iris_clasif_completo.csv
    └── dataset_estudio.csv
```

## 📖 Uso

1. **Cargar datos**: Usa el selector de archivos en la barra lateral para cargar tu dataset (CSV o XLSX)

2. **Explorar datos**: Revisa la vista previa y la información de las columnas

3. **Seleccionar algoritmo**: Elige entre:
   - Regresión Lineal Múltiple
   - Regresión Logística Binaria
   - K-Means

4. **Configurar variables**: Selecciona las columnas apropiadas para tu análisis

5. **Ejecutar modelo**: Haz clic en el botón de ejecución y revisa los resultados

6. **Analizar resultados**: Visualiza métricas y gráficos generados

## 📊 Datasets de Ejemplo

El proyecto incluye cuatro datasets de ejemplo en la carpeta `data/`:

- **multivar_dataset_calificacion.csv**: Para regresión lineal (predecir calificaciones)
- **logistic_dataset_educacion.csv**: Para regresión logística (clasificación binaria)
- **iris_clasif_completo.csv**: Para K-Means (clustering de flores iris)
- **dataset_estudio**: Para regresión lineal (predecir calificaciones [modificado porque el anterior no tiene mucha relacion entre si los valores del dataset])

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para la interfaz web
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Scikit-learn**: Algoritmos de machine learning
- **Matplotlib & Seaborn**: Visualización de datos

## 📝 Notas

- La aplicación maneja automáticamente valores nulos eliminando las filas afectadas
- Los datos se normalizan automáticamente cuando es necesario (K-Means)
- Las visualizaciones se generan dinámicamente según los resultados
- Puedes reiniciar la sesión en cualquier momento usando el botón en la barra lateral

## 🫱🏻‍🫲🏻 Contribuciones

Este es un proyecto educativo. Siéntete libre de modificarlo y adaptarlo a tus necesidades.

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## 🔝 Colaboradores

- **Garcia Covarrubias Alejandro Jesús**
- **Loza Sandoval Leonardo Sebastian**
- **Martinez Aviña Roberto Carlos**

#### **Estudiantes del Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI)**
#### **Universidad de Guadalajara**
