"""
Programa de Análisis CanSat - Versión Profesional
Autor: Carlos
Fecha: [Fecha]
Licencia: GPL-3.0
"""

# ======================
# Importación de librerías
# ======================
import os
import sys
import logging
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from logging.handlers import RotatingFileHandler

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
    """Configura el sistema de registro de eventos profesional"""
    
    logger = logging.getLogger('cansat')
    logger.setLevel(logging.DEBUG)

    # Formato profesional con colores (Windows compatible)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Archivo rotativo (1 MB por archivo, 3 backups)
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

# ======================
# Módulo de datos
# ======================
def cargar_datos(file_path: str) -> pd.DataFrame:
    """
    Carga y valida los datos del CanSat desde un archivo CSV
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con datos procesados
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        pd.errors.EmptyDataError: Si el archivo está vacío
        KeyError: Si faltan columnas esenciales
    """
    logger = logging.getLogger('cansat')
    
    try:
        # Verificación básica del archivo
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
        if os.path.getsize(file_path) == 0:
            raise pd.errors.EmptyDataError("El archivo CSV está vacío")

        # Carga de datos
        logger.info(f"Cargando datos desde: {file_path}")
        df = pd.read_csv(
            file_path,
            names=COLUMNAS_ESPERADAS,
            delimiter=',',
            skipinitialspace=True
        )
        
        # Validación de estructura
        for col in ['Altitude', 'Pressure', 'Temperature']:
            if col not in df.columns:
                raise KeyError(f"Columna crítica faltante: {col}")

        # Conversión segura de tipos de datos
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

        # Conversión de tiempo
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        except ValueError:
            logger.warning("Formato de tiempo no válido, usando valores crudos")
            df['Time'] = df['Time'].astype(str)

        logger.info(f"Datos cargados correctamente. Muestras: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}", exc_info=True)
        raise

# ======================
# Módulo de visualización
# ======================
def generar_graficas_interactivas(df: pd.DataFrame) -> dict:
    """
    Genera un conjunto de visualizaciones interactivas profesionales
    
    Args:
        df (pd.DataFrame): DataFrame con datos procesados
        
    Returns:
        dict: Diccionario con figuras de Plotly
    """
    logger = logging.getLogger('cansat')
    figuras = {}
    
    try:
        # --------------------------------------------------
        # Gráfico 1: Evolución temporal de altitud y presión
        # --------------------------------------------------
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Altitud
        fig1.add_trace(
            px.line(df, x='Index', y='Altitude').data[0],
            secondary_y=False
        )
        
        # Presión
        fig1.add_trace(
            px.line(df, x='Index', y='Pressure', color_discrete_sequence=['red']).data[0],
            secondary_y=True
        )
        
        fig1.update_layout(
            title='Dual Axis: Altitud y Presión vs Tiempo',
            xaxis_title='Índice de Muestreo',
            yaxis_title='Altitud (m)',
            yaxis2_title='Presión (hPa)',
            hovermode='x unified'
        )
        figuras['altitud_presion'] = fig1
        
        # --------------------------------------------------
        # Gráfico 2: Mapa 3D de trayectoria con sensores con imagen de fondo  
        # --------------------------------------------------
        fig2 = px.scatter_3d(
            df,
            x='GPS_Lon',
            y='GPS_Lat',
            z='Altitude',
            color='Temperature',
            hover_data=['Pressure', 'Time'],
            title='Trayectoria 3D del Vuelo',
            labels={
                'GPS_Lon': 'Longitud',
                'GPS_Lat': 'Latitud',
                'Altitude': 'Altitud (m)',
                'Temperature': 'Temp (°C)'
            }
        )
        
        figuras['trayectoria_3d'] = fig2
        
        # --------------------------------------------------
        # Gráfico 3: Panel de diagnóstico de sensores
        # --------------------------------------------------
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Acelerómetro X',
                'Acelerómetro Y',
                'Acelerómetro Z',
                'Distribución de Temperaturas'
            )
        )
        
        # Acelerómetros
        for i, col in enumerate(['AccX', 'AccY', 'AccZ'], 1):
            fig3.add_trace(
                px.line(df, x='Index', y=col).data[0],
                row=(i//2)+1, col=(i%2)+1
            )
            
        # Histograma de temperatura
        fig3.add_trace(
            px.histogram(df, x='Temperature', nbins=20).data[0],
            row=2, col=2
        )
        
        fig3.update_layout(
            title_text='Panel de Diagnóstico de Sensores',
            height=800,
            showlegend=False
        )
        figuras['panel_sensores'] = fig3
        
        logger.info("Gráficos generados exitosamente")
        return figuras
    
    except Exception as e:
        logger.error(f"Error generando gráficos: {str(e)}", exc_info=True)
        raise

# ======================
# Función principal
# ======================
def main():
    """Punto de entrada principal del programa"""
    
    logger = configurar_logging()
    
    try:
        logger.info("==== INICIO DEL PROCESO ====")
        
        # 1. Carga de datos
        df = cargar_datos('datos.csv')
        
        # 2. Generación de gráficos
        figuras = generar_graficas_interactivas(df)
        
        # 3. Exportación de resultados
        os.makedirs('graficos', exist_ok=True)
        
        for nombre, figura in figuras.items():
            figura.write_html(f"graficos/{nombre}.html")
            logger.info(f"Gráfico guardado: graficos/{nombre}.html")
            
        logger.info("==== PROCESO COMPLETADO ====")
        
    except FileNotFoundError as e:
        logger.error(f"Error crítico: {str(e)}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("El archivo CSV está vacío o corrupto")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error no controlado: {str(e)}", exc_info=True)
        sys.exit(1)

# ======================
# Punto de entrada
# ======================
if __name__ == "__main__":
    main()

