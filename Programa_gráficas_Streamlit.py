"""
Análisis CanSat - Versión Integral con Streamlit
Autor: Carlos
Licencia: GPL-3.0
"""

import os
import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from logging.handlers import RotatingFileHandler
import folium
import streamlit.components.v1 as components

# ======================
# Configuración global
# ======================
COLUMNAS_ESPERADAS = [
    "Tag", "Index", "Temperature", "Pressure", "Altitude",
    "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
    "GPS_Lat", "GPS_Lon", "Date", "Time"
]

# ======================
# Configuración de logging
# ======================
def configurar_logging():
    logger = logging.getLogger('cansat')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        'cansat.log',
        maxBytes=1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = configurar_logging()

# ======================
# Función para cargar y validar datos
# ======================
def cargar_datos(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, names=COLUMNAS_ESPERADAS, delimiter=',', skipinitialspace=True)
        
        # Validación de columnas críticas
        for col in ['Altitude', 'Pressure', 'Temperature']:
            if col not in df.columns:
                raise KeyError(f"Columna crítica faltante: {col}")
        
        conversiones = {
            'Index': ('int', 0),
            'Altitude': ('float', 0.0),
            'Pressure': ('float', 1013.25),
            'Temperature': ('float', 15.0),
            'GPS_Lat': ('float', 0.0),
            'GPS_Lon': ('float', 0.0)
        }
        
        for col, (dtype, default) in conversiones.items():
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
                if dtype == 'int':
                    df[col] = df[col].astype(int)
            except Exception as e:
                logger.warning(f"Error en columna {col}: {str(e)}")
                df[col] = default

        # Conversión de la columna de tiempo
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        except Exception as e:
            logger.warning("Formato de tiempo no válido, usando valores crudos")
            df['Time'] = df['Time'].astype(str)

        logger.info(f"Datos cargados correctamente. Muestras: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}", exc_info=True)
        st.error(f"Error al cargar datos: {e}")
        return None

# ======================
# Función: Estadísticas Descriptivas
# ======================
def analisis_descriptivo(df: pd.DataFrame):
    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())

# ======================
# Función: Matriz de Correlación
# ======================
def analisis_correlacion(df: pd.DataFrame):
    st.subheader("Matriz de Correlación")
    corr = df.corr()
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Viridis'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ======================
# Función: Análisis Temporal y Series de Tiempo
# ======================
def analisis_temporal(df: pd.DataFrame):
    st.subheader("Análisis Temporal")
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            # Combinar columnas Date y Time en un solo timestamp
            df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].astype(str))
            df.sort_values('Timestamp', inplace=True)
            # Ejemplo: resample de temperatura cada 10 minutos
            df.set_index('Timestamp', inplace=True)
            df_resampled = df['Temperature'].resample('10T').mean()
            st.line_chart(df_resampled)
        except Exception as e:
            st.error(f"Error en el análisis temporal: {e}")
    else:
        st.warning("No se disponen de las columnas 'Date' y 'Time' para análisis temporal.")

# ======================
# Función: Análisis del Comportamiento de los Sensores
# ======================
def analisis_sensores(df: pd.DataFrame):
    st.subheader("Comportamiento de los Sensores")
    # Ejemplo: Gráfico de acelerómetro y su versión suavizada
    if 'AccX' in df.columns:
        df['AccX_smoothed'] = df['AccX'].rolling(window=5).mean()
        fig = px.line(df, x='Index', y=['AccX', 'AccX_smoothed'], 
                      labels={'value': 'Aceleración X', 'variable': 'Medición'},
                      title='Acelerómetro X y Suavizado (Media Móvil)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encuentra la columna 'AccX' para análisis de sensores.")

# ======================
# Función: Generar Mapa Geoespacial
# ======================
def generar_mapa_geoespacial(df: pd.DataFrame):
    st.subheader("Mapa Geoespacial")
    if df[['GPS_Lat', 'GPS_Lon']].isnull().any().any():
        st.warning("Existen puntos GPS sin datos. No se puede generar el mapa correctamente.")
        return None
    else:
        center = [df['GPS_Lat'].mean(), df['GPS_Lon'].mean()]
        m = folium.Map(location=center, zoom_start=12)
        folium.PolyLine(
            df[['GPS_Lat', 'GPS_Lon']].values,
            color="blue",
            weight=2.5,
            opacity=1
        ).add_to(m)
        return m

# ======================
# Función: Detección de Eventos
# ======================
def analisis_eventos(df: pd.DataFrame):
    st.subheader("Detección de Eventos")
    # Ejemplo: Detectar picos en aceleración (umbral arbitrario)
    if 'AccX' in df.columns:
        threshold = st.number_input("Umbral para detección de picos en AccX", value=5.0)
        df['Evento'] = df['AccX'] > threshold
        eventos = df[df['Evento']]
        st.write(f"Se detectaron {len(eventos)} eventos con AccX > {threshold}")
        st.dataframe(eventos[['Index', 'AccX']])
        # Visualizar eventos en un gráfico
        fig = px.scatter(df, x='Index', y='AccX', color='Evento',
                         title='Eventos detectados en AccX')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encuentra la columna 'AccX' para detección de eventos.")

# ======================
# Función principal para Streamlit
# ======================
def main():
    st.title("Análisis Integral CanSat con Streamlit")
    st.write("Sube el archivo CSV con los datos del CanSat para explorar diferentes análisis.")

    file = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    
    if file is not None:
        df = cargar_datos(file)
        if df is not None:
            st.success(f"Datos cargados correctamente. Muestras: {len(df)}")
            
            # Organización en pestañas para cada análisis
            tabs = st.tabs(["Datos Crudos", "Estadísticas", "Correlación", "Temporal", "Sensores", "Geoespacial", "Eventos"])
            
            # Pestaña 1: Datos Crudos
            with tabs[0]:
                st.subheader("Datos Crudos")
                st.dataframe(df)
            
            # Pestaña 2: Estadísticas Descriptivas
            with tabs[1]:
                analisis_descriptivo(df)
            
            # Pestaña 3: Matriz de Correlación
            with tabs[2]:
                analisis_correlacion(df)
            
            # Pestaña 4: Análisis Temporal
            with tabs[3]:
                analisis_temporal(df)
            
            # Pestaña 5: Comportamiento de Sensores
            with tabs[4]:
                analisis_sensores(df)
            
            # Pestaña 6: Mapa Geoespacial
            with tabs[5]:
                mapa = generar_mapa_geoespacial(df)
                if mapa is not None:
                    components.html(mapa._repr_html_(), width=700, height=500)
            
            # Pestaña 7: Detección de Eventos
            with tabs[6]:
                analisis_eventos(df)
        else:
            st.error("No se pudo cargar el archivo CSV correctamente.")

if __name__ == "__main__":
    main()
