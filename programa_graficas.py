import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os

# ========================================
# Módulo de procesamiento de datos
# ========================================

def cargar_datos(file_path):
    """
    Carga los datos del CanSat desde un archivo CSV
    
    Parámetros:
        file_path (str): Ruta del archivo CSV
        
    Retorna:
        pd.DataFrame: DataFrame con los datos procesados
    """
    try:
        columns = [
            "Tag", "Index", "Temperature", "Pressure", "Altitude",
            "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
            "GPS_Lat", "GPS_Lon", "Date", "Time"
        ]
        
        df = pd.read_csv(file_path, names=columns, delimiter=",", skipinitialspace=True)
        
        # Conversión de tipos de datos con manejo de errores
        conversions = {
            "Index": int,
            "Temperature": float,
            "Pressure": float,
            "Altitude": float,
            "Time": lambda x: pd.to_datetime(x, format='%H:%M:%S', errors='coerce'),
            "GPS_Lat": lambda x: pd.to_numeric(x, errors='coerce'),
            "GPS_Lon": lambda x: pd.to_numeric(x, errors='coerce')
        }
        
        return df.apply(lambda col: col.map(conversions.get(col.name, lambda x: x)))
    
    except Exception as e:
        print(f"Error cargando datos: {str(e)}")
        return None

# ========================================
# Módulo de visualización completo
# ========================================

def generar_graficas_completas(df):
    """Genera todas las visualizaciones clave del dataset"""
    
    # Configuración estética común
    plt.style.use('seaborn-v0_8-darkgrid')
    palette = sns.color_palette("husl", 8)
    
    # 1. Gráfico de Altitud vs Tiempo
    plt.figure(figsize=(14, 6))
    sns.lineplot(x="Index", y="Altitude", data=df, 
                linewidth=1.5, marker="o", markersize=6,
                color=palette[0], label="Altitud")
    plt.title("Evolución Temporal de la Altitud", fontsize=14, pad=20)
    plt.xlabel("Índice de Muestreo", fontsize=12)
    plt.ylabel("Altitud (m)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 2. Presión vs Altitud con Temperatura
    plt.figure(figsize=(14, 6))
    scatter = sns.scatterplot(x="Altitude", y="Pressure", data=df,
                            hue="Temperature", palette="viridis",
                            s=100, edgecolor='black')
    plt.title("Relación Presión-Altitud con Temperatura", fontsize=14, pad=20)
    plt.xlabel("Altitud (m)", fontsize=12)
    plt.ylabel("Presión (Pa)", fontsize=12)
    plt.legend(title="Temp (°C)", title_fontsize=10, fontsize=9)
    plt.colorbar(scatter.collections[0], label="Temperatura (°C)")
    plt.tight_layout()
    plt.show()

    # 3. Aceleraciones en 3 Ejes
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    sns.lineplot(x="Index", y="AccX", data=df, color=palette[2], label="AccX")
    plt.ylabel("Aceleración X (g)", fontsize=10)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 2)
    sns.lineplot(x="Index", y="AccY", data=df, color=palette[3], label="AccY")
    plt.ylabel("Aceleración Y (g)", fontsize=10)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    sns.lineplot(x="Index", y="AccZ", data=df, color=palette[4], label="AccZ")
    plt.ylabel("Aceleración Z (g)", fontsize=10)
    plt.legend(loc='upper right')

    plt.suptitle("Aceleraciones en los Tres Ejes", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # 4. Temperatura vs Tiempo con Regresión
    plt.figure(figsize=(14, 6))
    sns.regplot(x="Index", y="Temperature", data=df, 
                scatter_kws={'s': 80, 'alpha': 0.6},
                line_kws={'color': 'red', 'linestyle': '--'},
                label="Tendencia")
    sns.lineplot(x="Index", y="Temperature", data=df, 
                color=palette[5], alpha=0.5, label="Temperatura")
    plt.title("Evolución de la Temperatura con Tendencia", fontsize=14, pad=20)
    plt.xlabel("Índice de Muestreo", fontsize=12)
    plt.ylabel("Temperatura (°C)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 5. Gráfico 3D de Aceleraciones (Nueva!)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(df["AccX"], df["AccY"], df["AccZ"],
                   c=df["Altitude"], cmap="plasma",
                   s=50, alpha=0.7)
    
    ax.set_xlabel("AccX (g)", fontsize=10)
    ax.set_ylabel("AccY (g)", fontsize=10)
    ax.set_zlabel("AccZ (g)", fontsize=10)
    ax.set_title("Espacio 3D de Aceleraciones Coloreado por Altitud", pad=20)
    
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label("Altitud (m)", fontsize=10)
    
    plt.tight_layout()
    plt.show()

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
# Función Principal
# ========================================

def main():
    # Configuración
    archivo_csv = "datos.csv"
    archivo_kml = "ruta_cansat.kml"
    
    # Verificar existencia de archivo
    if not os.path.exists(archivo_csv):
        print(f"Error: No se encuentra el archivo {archivo_csv}")
        return
    
    # Procesamiento de datos
    df = cargar_datos(archivo_csv)
    
    if df is None:
        return
    
    # Visualización completa
    generar_graficas_completas(df)
    
    # Generación KML
    datos_gps = leer_datos_csv(archivo_csv)
    crear_kml_mejorado(datos_gps, archivo_kml)
    print(f"KML generado exitosamente: {archivo_kml}")

# ========================================
# Funciones Auxiliares (mantenidas de tu versión original)
# ========================================

def leer_datos_csv(archivo_csv):
    datos_gps = []
    with open(archivo_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "JADA":
                datos_gps.append(row)
    return datos_gps

# Ejecución del programa
if __name__ == "__main__":
    main()
