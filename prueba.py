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
import csv
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
        )
        figuras['altitud_presion'] = fig1

        # Gráfico 2: Trayectoria 3D
        fig2 = px.scatter_3d(df, x='GPS_Lon', y='GPS_Lat', z='Altitude', color='Temperature',
                             hover_data=['Pressure', 'Time'],
                             title='Trayectoria 3D del Vuelo',
                             labels={'GPS_Lon': 'Longitud', 'GPS_Lat': 'Latitud', 'Altitude': 'Altitud (m)', 'Temperature': 'Temp (°C)'})
        figuras['trayectoria_3d'] = fig2

        # Gráfico 3: Panel sensores
        fig3 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=('Acelerómetro X', 'Acelerómetro Y', 'Acelerómetro Z', 'Distribución de Temperaturas')
        )

        # Acelerómetro X
        trace_accx = px.line(df, x='Index', y='AccX', labels={'x': 'Tiempo (índice)', 'y': 'Aceleración X (m/s²)'}).data[0]
        fig3.add_trace(trace_accx, row=1, col=1)

        # Acelerómetro Y
        trace_accy = px.line(df, x='Index', y='AccY', labels={'x': 'Tiempo (índice)', 'y': 'Aceleración Y (m/s²)'}).data[0]
        fig3.add_trace(trace_accy, row=1, col=2)

        # Acelerómetro Z
        trace_accz = px.line(df, x='Index', y='AccZ', labels={'x': 'Tiempo (índice)', 'y': 'Aceleración Z (m/s²)'}).data[0]
        fig3.add_trace(trace_accz, row=2, col=1)

        # Histograma de Temperatura
        trace_temp = px.histogram(df, x='Temperature', nbins=20, labels={'x': 'Temperatura (°C)', 'y': 'Frecuencia'}).data[0]
        fig3.add_trace(trace_temp, row=2, col=2)

        # Ajustes de diseño y explicación
        fig3.update_layout(
            title_text='Panel de Diagnóstico de Sensores',
            height=800,
            showlegend=False,
            #color titles
            font=dict(size=12, color="#a40000"),
        )

        figuras['panel_sensores'] = fig3


        # Gráfico 4: Matriz de correlación
        variables = ['Temperature', 'Pressure', 'Altitude', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'GPS_Lat', 'GPS_Lon']

        corr_matrix = df[variables].corr()

        fig4 = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            labels=dict(color="Coef. de correlación")
        )

        # Añadir etiquetas claras en los ejes y ajustar el diseño
        fig4.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables",
            margin=dict(t=150),  # aumentar espacio superior
        )

        figuras['matriz_correlacion'] = fig4


        # Gráfico 5: Diagramas de dispersión
        fig5a = px.scatter(df, x='Altitude', y='Pressure', trendline='ols', title="Altitud vs Presión")
        fig5b = px.scatter(df, x='Altitude', y='Temperature', trendline='ols', title="Altitud vs Temperatura")
        fig5c = px.scatter(df, x='Temperature', y='Pressure', trendline='ols', title="Temperatura vs Presión")

        figuras['scatter_altitud_presion'] = fig5a
        figuras['scatter_altitud_temp'] = fig5b
        figuras['scatter_temp_presion'] = fig5c

        logger.info("Gráficos generados exitosamente")
        return figuras

    except Exception as e:
        logger.error(f"Error generando gráficos: {str(e)}", exc_info=True)
        raise

# ======================
# Módulo de añadir logotipo
# ======================
def insertar_encabezado_logo(html_path: str, logo_svg: str, nombre_equipo: str, titulo_grafico: str = "", descripcion: str = ""):
    """
    Inserta un encabezado con logotipo, nombre del equipo, título y descripción opcional.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        contenido = f.read()
 
    encabezado = f"""
    <header style="display:flex;align-items:center;border-bottom:2px solid #ccc;padding:20px;margin-bottom:20px;">
        <!-- Contenedor del logo -->
        <div style="flex-shrink: 0; margin-right: 20px;">
            <img src="{logo_svg}" alt="Logo {nombre_equipo}" style="height:80px;">
        </div>
        
        <!-- Contenedor del título y descripción -->
        <div style="max-width:80%;">
            <h2 style="margin:0;font-size:1.4em;text-align:left;color: a40000;">{titulo_grafico}</h2>
            <p style="margin-top:10px;line-height:1.4em;text-align:left;">
                {descripcion}
            </p>
        </div>
    </header>
    """

    contenido_modificado = contenido.replace("<body>", f"<body>\n{encabezado}")

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(contenido_modificado)

    print(f"✅ Encabezado insertado en {html_path}")


# ========================================
# Módulo de generación KML
# ========================================

def crear_kml_mejorado(datos_gps, archivo_kml):
    """
    Crea un archivo KML con trayectoria continua y puntos destacados
    
    Parámetros:
        datos_gps (list): Lista de tuplas con datos GPS
        archivo_kml (str): Ruta de salida para el archivo KML
    """
    kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>Trayectoria CanSat</name>
        <description>Vuelo completo con datos de sensores</description>
        
        <!-- Estilo para la ruta -->
        <Style id="ruta_estilo">
            <LineStyle>
                <color>ff00aaff</color>
                <width>4</width>
            </LineStyle>
        </Style>
        
        <!-- Trayectoria completa -->
        <Placemark>
            <name>Ruta Completa</name>
            <styleUrl>#ruta_estilo</styleUrl>
            <LineString>
                <extrude>1</extrude>
                <altitudeMode>absolute</altitudeMode>
                <coordinates>
                {coordenadas}
                </coordinates>
            </LineString>
        </Placemark>
        
        <!-- Puntos de muestreo -->
        {puntos}
    </Document>
    </kml>'''
    
    # Generar coordenadas y puntos
    coordenadas = []
    puntos_kml = ""
    
    for i, row in enumerate(datos_gps):
        try:
            lat = float(row[11])  # Índices ajustados según tu CSV
            lon = float(row[12])
            alt = float(row[4])
            
            coordenadas.append(f"{lon},{lat},{alt}")
            
            puntos_kml += f'''
            <Placemark>
                <name>Muestra {i}</name>
                <description>
                    Altitud: {alt:.2f} m
                    Presión: {float(row[3]):.2f} Pa
                    Temp: {float(row[2]):.2f} °C
                </description>
                <Point>
                    <coordinates>{lon},{lat},{alt}</coordinates>
                </Point>
            </Placemark>'''
            
        except (ValueError, IndexError):
            continue
    
    kml_final = kml_template.format(
        coordenadas="\n".join(coordenadas),
        puntos=puntos_kml
    )
    
    with open(archivo_kml, 'w', encoding='utf-8') as f:
        f.write(kml_final)

# ========================================
# Módulo de carga de datos
# ========================================

def leer_datos_csv(archivo_csv):
    datos_gps = []
    with open(archivo_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "JADA":
                datos_gps.append(row)
    return datos_gps

# ========================================
# Función principal
# ========================================

def main():
    """Punto de entrada principal del programa"""
    
    logger = configurar_logging()
    titulos = {
    "grafico_temperatura_altura": "Evolución de Temperatura y Altura",
    "trayectoria_3d": "Trayectoria 3D del Vuelo y temperatura",
    "panel_sensores": "Panel de Diagnóstico de Sensores",
    "matriz_correlacion": "Matriz de Correlación entre Variables",
    "altitud_presion": "Altitud y Presión vs Paquete",
    "scatter_altitud_presion": "Altitud vs Presión",
    "scatter_altitud_temp": "Altitud vs Temperatura",
    "scatter_temp_presion": "Temperatura vs Presión",
    
    }
    descripciones = {
    "altitud_presion": (
        "Este gráfico muestra la relación entre la altitud y la presión durante el vuelo. "
        "Se observa cómo la altitud aumenta a medida que el CanSat asciende y cómo la presión disminuye a medida que aumenta la altitud."
    ),
    "trayectoria_3d": (
        "La trayectoria 3D muestra el recorrido del CanSat durante el vuelo. "
        "Cada punto en el gráfico representa la posición geográfica del CanSat junto con su altitud y temperatura en ese momento."
    ),
    "matriz_correlacion": (
        "Esta matriz muestra cómo se relacionan las distintas variables físicas registradas por el CanSat. "
        "Valores cercanos a +1 (rojo) indican correlación positiva fuerte, valores cercanos a -1 (azul) indican correlación negativa fuerte, "
        "y valores cercanos a 0 (blanco) indican que no hay relación lineal."
    ),
    "panel_sensores": (
        "Este panel muestra cómo varía la aceleración en los ejes X, Y y Z del CanSat a lo largo del tiempo, "
        "así como la distribución de las temperaturas registradas durante el vuelo. "
        "Los acelerómetros permiten analizar el movimiento y la orientación del satélite."
    ),
    "altitud_tiempo": (
        "Representación de la altitud del CanSat a lo largo del tiempo. Muestra claramente las fases de ascenso, caída y aterrizaje."
    ),
    "scatter_altitud_presion": (
        "Este gráfico muestra la relación entre la altitud y la presión. "
        "Se puede observar cómo la presión disminuye conforme aumenta la altitud."
    ),
    "scatter_altitud_temp": (
        "Este gráfico muestra la relación entre la altitud y la temperatura. "
        "A medida que el CanSat asciende, la temperatura puede variar de acuerdo con la altitud."
    ),
    "scatter_temp_presion": (
        "Este gráfico muestra cómo la temperatura y la presión están relacionadas. "
        "Generalmente, la temperatura y la presión se afectan entre sí, con cambios notables a diferentes altitudes."
    ),
}
    try:
        logger.info("==== INICIO DEL PROCESO ====")
        
        # 1. Carga de datos
        df = cargar_datos('datos.csv')
        archivo_kml = "graficos/ruta_cansat.kml"
        
        # 2. Generación de gráficos
        figuras = generar_graficas_interactivas(df)
        
        # 3. Exportación de resultados
        os.makedirs('graficos', exist_ok=True)
        
        for nombre, figura in figuras.items():
            ruta_html = f"graficos/{nombre}.html"
            figura.update_layout(title=None)  # Elimina título interno si molesta
            figura.write_html(ruta_html)
            insertar_encabezado_logo(
                html_path=ruta_html,
                logo_svg="JADA CANSAT TEAM.svg",
                nombre_equipo="JADA CanSat Team",
                titulo_grafico=titulos.get(nombre, ""),
                descripcion=descripciones.get(nombre, "")
            )
            logger.info(f"Gráfico guardado: {ruta_html}")
        
        # Generación KML
        datos_gps = leer_datos_csv('datos.csv')
        crear_kml_mejorado(datos_gps, archivo_kml)
        print(f"KML generado exitosamente: {archivo_kml}")

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