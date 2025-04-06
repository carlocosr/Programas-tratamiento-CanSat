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
        # Gráfico 1: Altitud y presión
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(px.line(df, x='Index', y='Altitude').data[0], secondary_y=False)
        fig1.add_trace(px.line(df, x='Index', y='Pressure', color_discrete_sequence=['red']).data[0], secondary_y=True)
        fig1.update_layout(
            title='Dual Axis: Altitud y Presión vs Tiempo',
            xaxis_title='Índice de Muestreo',
            yaxis_title='Altitud (m)',
            yaxis2_title='Presión (hPa)',
            hovermode='x unified',
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Este gráfico muestra la relación entre la altitud y la presión durante el vuelo. ' +
                         'Se observa cómo la altitud aumenta a medida que el CanSat asciende y cómo la presión disminuye a medida que aumenta la altitud.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        figuras['altitud_presion'] = fig1

        # Gráfico 2: Trayectoria 3D
        fig2 = px.scatter_3d(df, x='GPS_Lon', y='GPS_Lat', z='Altitude', color='Temperature',
                             hover_data=['Pressure', 'Time'],
                             title='Trayectoria 3D del Vuelo',
                             labels={'GPS_Lon': 'Longitud', 'GPS_Lat': 'Latitud', 'Altitude': 'Altitud (m)', 'Temperature': 'Temp (°C)'})
        fig2.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='La trayectoria 3D muestra el recorrido del CanSat durante el vuelo. ' +
                         'Cada punto en el gráfico representa la posición geográfica del CanSat junto con su altitud y temperatura en ese momento.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        figuras['trayectoria_3d'] = fig2

        # Gráfico 3: Panel sensores
        fig3 = make_subplots(rows=2, cols=2,
                             subplot_titles=('Acelerómetro X', 'Acelerómetro Y', 'Acelerómetro Z', 'Distribución de Temperaturas'))
        for i, col in enumerate(['AccX', 'AccY', 'AccZ'], 1):
            fig3.add_trace(px.line(df, x='Index', y=col).data[0], row=(i//2)+1, col=(i%2)+1)
        fig3.add_trace(px.histogram(df, x='Temperature', nbins=20).data[0], row=2, col=2)
        fig3.update_layout(
            title_text='Panel de Diagnóstico de Sensores',
            height=800,
            showlegend=False,
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Este panel muestra la lectura de tres acelerómetros en los ejes X, Y, Z, ' +
                         'además de la distribución de las temperaturas registradas durante el vuelo. ' +
                         'Cada acelerómetro mide la aceleración en uno de los ejes espaciales.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        figuras['panel_sensores'] = fig3

        # Gráfico 4: Matriz de correlación
        variables = ['Temperature', 'Pressure', 'Altitude', 'AccX', 'AccY', 'AccZ']
        corr_matrix = df[variables].corr()
        fig4 = px.imshow(corr_matrix,
                         text_auto=True,
                         color_continuous_scale='RdBu',
                         title="Matriz de Correlación de Variables Clave")
        fig4.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='La matriz de correlación muestra cómo las diferentes variables del CanSat se relacionan entre sí. ' +
                         'Valores cercanos a +1 o -1 indican una relación fuerte, mientras que valores cercanos a 0 indican una relación débil o nula.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        figuras['matriz_correlacion'] = fig4

        # Gráfico 5: Diagramas de dispersión
        fig5a = px.scatter(df, x='Altitude', y='Pressure', trendline='ols', title="Altitud vs Presión")
        fig5b = px.scatter(df, x='Altitude', y='Temperature', trendline='ols', title="Altitud vs Temperatura")
        fig5c = px.scatter(df, x='Temperature', y='Pressure', trendline='ols', title="Temperatura vs Presión")
        fig5a.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Este gráfico muestra la relación entre la altitud y la presión. ' +
                         'Se puede observar cómo la presión disminuye conforme aumenta la altitud.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        fig5b.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Este gráfico muestra la relación entre la altitud y la temperatura. ' +
                         'A medida que el CanSat asciende, la temperatura puede variar de acuerdo con la altitud.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        fig5c.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Este gráfico muestra cómo la temperatura y la presión están relacionadas. ' +
                         'Generalmente, la temperatura y la presión se afectan entre sí, con cambios notables a diferentes altitudes.',
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    align="center"
                )
            ]
        )
        figuras['scatter_altitud_presion'] = fig5a
        figuras['scatter_altitud_temp'] = fig5b
        figuras['scatter_temp_presion'] = fig5c

        logger.info("Gráficos generados exitosamente")
        return figuras

    except Exception as e:
        logger.error(f"Error generando gráficos: {str(e)}", exc_info=True)
        raise



# ======================
# Función principal
# ======================
def insertar_encabezado_logo(html_path: str, logo_svg: str, nombre_equipo: str):
    """
    Inserta un encabezado con logotipo SVG y el nombre del equipo al principio del HTML generado.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        contenido = f.read()

    encabezado = f"""
    <header style="display:flex;align-items:center;border-bottom:2px solid #ccc;padding-bottom:10px;margin-bottom:20px;align-content: center;justify-content: center;">
        <img src="{logo_svg}" alt="Logo {nombre_equipo}" style="height:80px;margin-right:20px;">
    </header>
    """

    contenido_modificado = contenido.replace("<body>", f"<body>\n{encabezado}")

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(contenido_modificado)

    print(f"✅ Logotipo y encabezado insertados en {html_path}")

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
            ruta_html = f"graficos/{nombre}.html"
            figura.write_html(ruta_html)
            insertar_encabezado_logo(
                html_path=ruta_html,
                logo_svg="JADA CANSAT TEAM.svg",
                nombre_equipo="JADA CanSat Team"
            )
            logger.info(f"Gráfico guardado: {ruta_html}")
            
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

if __name__ == "__main__":
    main()