"""
Programa de Análisis CanSat - Versión Profesional Adaptada para Datos del Receptor
Autor: Carlos (Modificado)
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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logging.handlers import RotatingFileHandler

# ======================
# Configuración global
# ======================
# Actualizado para reflejar las columnas del receptor
COLUMNAS_ESPERADAS = [
    "id", "Paquete", "Temperatura", "Presion", "Altitud",
    "Latitud", "Longitud", "AccZ"
]

archivo = "datos_receptor.csv"  # Para el receptor

# ======================
# Configuración de logging
# ======================
def configurar_logging():
    """Configura el sistema de registro de eventos profesional"""
    
    # Creamos el objeto logger
    logger = logging.getLogger('cansat')
    # Establecemos el nivel de logging (debug, info, warning, error, critical)
    logger.setLevel(logging.DEBUG)

    # Formato profesional con colores (Windows compatible)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler()
    # Establecemos el nivel de logging para la consola
    console_handler.setLevel(logging.INFO)
    # Le damos formato al logger
    console_handler.setFormatter(formatter)

    # Archivo rotativo (1 MB por archivo, 3 backups)
    file_handler = RotatingFileHandler(
        'cansat.log',
        maxBytes=1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    # Establecemos el nivel de logging para el archivo
    file_handler.setLevel(logging.DEBUG)
    # Le damos formato al logger
    file_handler.setFormatter(formatter)

    # Añadimos los handlers al logger
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
        for col in ['Altitud', 'Presion', 'Temperatura']:
            if col not in df.columns:
                raise KeyError(f"Columna crítica faltante: {col}")

        # Conversión segura de tipos de datos - Adaptado para columnas del receptor
        conversiones = {
            'Paquete': ('int', 0),
            'Altitud': ('float', 0.0),
            'Presion': ('float', 1013.25),  # defecto 1013.25 mbar (presión atmosférica a nivel del mar)
            'Temperatura': ('float', 15.0),  # defecto 15.0 C (temperatura ambiente)
            'Latitud': ('float', 0.0),
            'Longitud': ('float', 0.0),
            'AccZ': ('float', 0.0),
        }
        
        for col, (dtype, default) in conversiones.items():
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
                if dtype == 'int':
                    df[col] = df[col].astype(int)
            except Exception as e:
                logger.warning(f"Error en columna {col}: {str(e)}")
                df[col] = default

        # No hay columna de Hora en el receptor, por lo que omitimos esa parte

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
    Genera un conjunto de visualizaciones interactivas profesionales adaptadas para los datos del receptor
    
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

        # Asegurarse de que las trazas tengan nombres para la leyenda
        trace_altitude = px.line(df, x='Paquete', y='Altitud').data[0]
        trace_altitude.update(name='Altitud')

        trace_pressure = px.line(df, x='Paquete', y='Presion', color_discrete_sequence=['red']).data[0]
        trace_pressure.update(name='Presión')

        # Añadir las trazas
        fig1.add_trace(trace_altitude, secondary_y=False)
        fig1.add_trace(trace_pressure, secondary_y=True)

        # Actualizar el diseño, asegurándose de que la leyenda sea visible
        fig1.update_layout(
            title='Dual Axis: Altitud y Presión vs Tiempo',
            xaxis_title='Índice de Muestreo',
            yaxis_title='Altitud (m)',
            yaxis=dict(color='blue'),  # Cambiar color del eje Y
            yaxis2_title='Presión (hPa)',
            yaxis2=dict(color='red'),  # Cambiar color del eje Y2
            hovermode='x unified',
            showlegend=True,  # Hacer la leyenda visible
        )

        # Guardar la figura
        figuras['altitud_presion'] = fig1

        # Gráfico 2: Trayectoria 3D
        fig2 = px.scatter_3d(df, x='Longitud', y='Latitud', z='Altitud', color='Temperatura',
                             hover_data=['Presion'],  # Eliminamos 'Hora' que no existe en el receptor
                             title='Trayectoria 3D del Vuelo',
                             labels={'Longitud': 'Longitud', 'Latitud': 'Latitud', 'Altitud': 'Altitud (m)', 'Temperatura': 'Temp (°C)'})
        figuras['trayectoria_3d'] = fig2

        # Gráfico 3: Panel sensores simplificado para datos del receptor
        # Solo tenemos AccZ, así que modificamos el panel
        fig3 = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Acelerómetro Z', 'Distribución de Temperaturas')
        )

        # Acelerómetro Z
        trace_accz = px.line(df, x='Paquete', y='AccZ').data[0]
        fig3.add_trace(trace_accz, row=1, col=1)

        # Histograma de Temperatura
        trace_temp = px.histogram(df, x='Temperatura', nbins=20).data[0]
        fig3.add_trace(trace_temp, row=2, col=1)

        # Ajustes de diseño y explicación
        fig3.update_layout(
            title_text='Panel de Diagnóstico de Sensores (Datos Receptor)',
            height=800,
            showlegend=False,
            font=dict(size=12, color="#a40000"),
        )

        # Establecer los títulos de los ejes
        fig3.update_xaxes(title_text="Tiempo (índice)", row=1, col=1)
        fig3.update_yaxes(title_text="Aceleración (m/s²)", row=1, col=1)

        fig3.update_xaxes(title_text="Temperatura (°C)", row=2, col=1)
        fig3.update_yaxes(title_text="Frecuencia", row=2, col=1)

        figuras['panel_sensores'] = fig3


        # Gráfico 4: Matriz de correlación (adaptada para columnas disponibles)
        # Solo muestra el triángulo inferior (elimina valores duplicados)
        variables = ['Temperatura', 'Presion', 'Altitud']
        corr_matrix = df[variables].corr()

        # Crear una máscara para el triángulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) # k=1 mantiene la diagonal principal

        # Aplicar la máscara (convertimos a NaN los valores que queremos ocultar)
        corr_matrix_masked = corr_matrix.copy()
        corr_matrix_masked.values[mask] = np.nan

        # Crear el gráfico con la matriz enmascarada
        fig4 = px.imshow(
            corr_matrix_masked,
            text_auto=True,
            color_continuous_scale='RdBu',
            labels=dict(color="Coef. de correlación"),
            zmin=-1, zmax=1  # Asegurar que la escala de color vaya de -1 a 1
        )

        # Añadir etiquetas claras en los ejes y ajustar el diseño
        fig4.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables",
            margin=dict(t=150),  # aumentar espacio superior
        )

        figuras['matriz_correlacion'] = fig4

        # Gráfico 5: Diagramas de dispersión
        fig5a = px.scatter(df, x='Altitud', y='Presion', trendline='ols', trendline_color_override='red', title="Altitud vs Presión", trendline_scope='overall')
        fig5b = px.scatter(df, x='Altitud', y='Temperatura', trendline='ols', trendline_color_override='red', title="Altitud vs Temperatura", trendline_scope='overall')
        fig5c = px.scatter(df, x='Temperatura', y='Presion', trendline='ols', trendline_color_override='red', title="Temperatura vs Presión", trendline_scope='overall')

        figuras['scatter_altitud_presion'] = fig5a
        figuras['scatter_altitud_temp'] = fig5b
        figuras['scatter_temp_presion'] = fig5c

        # Nuevo gráfico: Temperatura vs Tiempo
        fig6 = px.line(df, x='Paquete', y='Temperatura', title='Evolución de Temperatura')
        fig6.update_layout(
            xaxis_title='Índice de Muestreo',
            yaxis_title='Temperatura (°C)'
        )
        figuras['temperatura_tiempo'] = fig6

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
    Adaptado para trabajar con los datos del receptor
    
    Parámetros:
        datos_gps (list): Lista de tuplas con datos GPS
        archivo_kml (str): Ruta de salida para el archivo KML
    """
    kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>Trayectoria CanSat (Receptor)</name>
        <description>Vuelo completo con datos de sensores (Datos del Receptor)</description>
        
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
            # Índices ajustados para el formato del receptor
            lat = float(row[5])  # Latitud
            lon = float(row[6])  # Longitud
            alt = float(row[4])  # Altitud
            
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
            if len(row) > 0 and row[0] == "JADA":
                datos_gps.append(row)
    return datos_gps

# ========================================
# Función principal
# ========================================

def main():
    """Punto de entrada principal del programa"""
    logger = configurar_logging()
    titulos = {
        "temperatura_tiempo": "Evolución de Temperatura en el Tiempo",
        "trayectoria_3d": "Trayectoria 3D del Vuelo y temperatura",
        "panel_sensores": "Panel de Diagnóstico de Sensores (Datos Receptor)",
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
            "La trayectoria 3D muestra el recorrido del CanSat durante el vuelo a partir de los datos recibidos por el receptor. "
            "Cada punto en el gráfico representa la posición geográfica del CanSat junto con su altitud y temperatura en ese momento."
        ),
        "matriz_correlacion": (
            "Esta matriz muestra cómo se relacionan las distintas variables físicas registradas por el receptor. "
            "Valores cercanos a +1 (rojo) indican correlación positiva fuerte, valores cercanos a -1 (azul) indican correlación negativa fuerte, "
            "y valores cercanos a 0 (blanco) indican que no hay relación lineal."
        ),
        "panel_sensores": (
            "Este panel muestra cómo varía la aceleración en el eje Z del CanSat a lo largo del tiempo, "
            "así como la distribución de las temperaturas registradas durante el vuelo. "
            "Los datos son los recibidos por el receptor y no incluyen todos los ejes de aceleración."
        ),
        "temperatura_tiempo": (
            "Este gráfico muestra la evolución de la temperatura a lo largo del tiempo de vuelo. "
            "Permite observar los cambios térmicos experimentados por el CanSat durante las distintas fases del vuelo."
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
        logger.info("==== INICIO DEL PROCESO (DATOS RECEPTOR) ====")
        
        # 1. Carga de datos
        df = cargar_datos(archivo)
        archivo_kml = "graficos/ruta_cansat_receptor.kml"
        
        # 2. Generación de gráficos
        figuras = generar_graficas_interactivas(df)
        
        # 3. Exportación de resultados
        os.makedirs('graficos', exist_ok=True)
        
        for nombre, figura in figuras.items():
            ruta_html = f"graficos/{nombre}_receptor.html"
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
        datos_gps = leer_datos_csv(archivo)
        crear_kml_mejorado(datos_gps, archivo_kml)
        print(f"KML generado exitosamente: {archivo_kml}")

        logger.info("==== PROCESO COMPLETADO (DATOS RECEPTOR) ====")
        
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