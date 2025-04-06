import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import folium
from datetime import datetime
import seaborn as sns

# Configuración para visualizaciones más atractivas
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Función para cargar y preparar los datos
def cargar_datos(archivo_csv):
    # Definir nombres de columnas
    columnas = ["Tag", "Index", "Temperature", "Pressure", "Altitude",
                "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
                "GPS_Lat", "GPS_Lon", "Date", "Time"]
    
    # Cargar el CSV
    df = pd.read_csv(archivo_csv, names=columnas)
    
    # Combinar fecha y hora en un solo campo datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    return df

# Función para crear gráficos de series temporales
def graficar_series_temporales(df):
    # Crear una figura con subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Graficar temperatura
    axes[0].plot(df['DateTime'], df['Temperature'], 'r-', linewidth=2)
    axes[0].set_ylabel('Temperatura (°C)')
    axes[0].set_title('Temperatura vs Tiempo')
    axes[0].grid(True)
    
    # Graficar presión
    axes[1].plot(df['DateTime'], df['Pressure'], 'b-', linewidth=2)
    axes[1].set_ylabel('Presión (Pa)')
    axes[1].set_title('Presión Atmosférica vs Tiempo')
    axes[1].grid(True)
    
    # Graficar altitud
    axes[2].plot(df['DateTime'], df['Altitude'], 'g-', linewidth=2)
    axes[2].set_ylabel('Altitud (m)')
    axes[2].set_xlabel('Tiempo')
    axes[2].set_title('Altitud vs Tiempo')
    axes[2].grid(True)
    
    # Formatear eje x para mejor visualización de tiempo
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes[2].xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

# Función para visualizar datos del acelerómetro y giroscopio
def graficar_movimiento(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Gráfico de acelerómetro
    axes[0].plot(df['DateTime'], df['AccX'], 'r-', label='X')
    axes[0].plot(df['DateTime'], df['AccY'], 'g-', label='Y')
    axes[0].plot(df['DateTime'], df['AccZ'], 'b-', label='Z')
    axes[0].set_ylabel('Aceleración')
    axes[0].set_title('Datos del Acelerómetro')
    axes[0].legend()
    axes[0].grid(True)
    
    # Gráfico del giroscopio
    axes[1].plot(df['DateTime'], df['GyroX'], 'r-', label='X')
    axes[1].plot(df['DateTime'], df['GyroY'], 'g-', label='Y')
    axes[1].plot(df['DateTime'], df['GyroZ'], 'b-', label='Z')
    axes[1].set_ylabel('Velocidad Angular')
    axes[1].set_xlabel('Tiempo')
    axes[1].set_title('Datos del Giroscopio')
    axes[1].legend()
    axes[1].grid(True)
    
    # Formatear eje x para mejor visualización de tiempo
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes[1].xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

# Función para crear un mapa con la trayectoria GPS
def crear_mapa_trayectoria(df):
    # Calcular el punto central para el mapa
    lat_media = df['GPS_Lat'].mean()
    lon_media = df['GPS_Lon'].mean()
    
    # Crear un mapa centrado en la ubicación media
    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=18)
    
    # Añadir marcadores para el inicio y fin
    folium.Marker(
        [df['GPS_Lat'].iloc[0], df['GPS_Lon'].iloc[0]],
        popup="Inicio",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(mapa)
    
    folium.Marker(
        [df['GPS_Lat'].iloc[-1], df['GPS_Lon'].iloc[-1]],
        popup="Fin",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(mapa)
    
    # Crear una línea para la trayectoria
    puntos = [[row['GPS_Lat'], row['GPS_Lon']] for _, row in df.iterrows()]
    folium.PolyLine(
        puntos,
        color="blue",
        weight=5,
        opacity=0.7
    ).add_to(mapa)
    
    # Guardar el mapa como HTML
    mapa.save("trayectoria_cansat.html")
    
    return mapa

# Función para crear un gráfico 3D de la trayectoria
def graficar_trayectoria_3d(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalizar las coordenadas para mejor visualización
    lon_ref = df['GPS_Lon'].iloc[0]
    lat_ref = df['GPS_Lat'].iloc[0]
    
    # Convertir a metros aproximados (muy aproximado)
    x = (df['GPS_Lon'] - lon_ref) * 85000  # ~85km por grado en longitud a estas latitudes
    y = (df['GPS_Lat'] - lat_ref) * 111000  # ~111km por grado en latitud
    z = df['Altitude']
    
    # Colorear según el tiempo (índice)
    colores = df['Index'] - df['Index'].min()
    
    # Trazar la trayectoria 3D
    scatter = ax.scatter(x, y, z, c=colores, cmap='viridis', s=30)
    ax.plot(x, y, z, 'r-', linewidth=1, alpha=0.3)
    
    # Etiquetas
    ax.set_xlabel('Longitud (m)')
    ax.set_ylabel('Latitud (m)')
    ax.set_zlabel('Altitud (m)')
    ax.set_title('Trayectoria 3D del CANSAT')
    
    # Añadir barra de colores
    cbar = plt.colorbar(scatter)
    cbar.set_label('Índice de tiempo')
    
    return fig

# Función para analizar correlaciones
def analizar_correlaciones(df):
    # Seleccionar solo variables numéricas relevantes
    vars_numericas = ['Temperature', 'Pressure', 'Altitude', 
                      'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    
    # Crear mapa de calor de correlaciones
    plt.figure(figsize=(12, 10))
    corr_matrix = df[vars_numericas].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                          cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación de Variables del CANSAT")
    plt.tight_layout()
    
    return plt.gcf()

# Ejemplo de uso con datos de ejemplo
def main():
    # Nota: Cambia 'datos_cansat.csv' al nombre de tu archivo CSV
    nombre_archivo = 'datos_cansat.csv'
    
    # Crear archivo de ejemplo si quieres probar el código sin tener los datos reales
    crear_archivo_ejemplo(nombre_archivo)
    
    # Cargar datos
    df = cargar_datos(nombre_archivo)
    
    # Mostrar información básica
    print("Primeras filas del dataset:")
    print(df.head())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Generar visualizaciones
    fig_series = graficar_series_temporales(df)
    fig_series.savefig('series_temporales.png')
    
    fig_movimiento = graficar_movimiento(df)
    fig_movimiento.savefig('datos_movimiento.png')
    
    mapa = crear_mapa_trayectoria(df)
    
    fig_3d = graficar_trayectoria_3d(df)
    fig_3d.savefig('trayectoria_3d.png')
    
    fig_corr = analizar_correlaciones(df)
    fig_corr.savefig('correlaciones.png')
    
    print("\nVisualización completada. Se han generado archivos PNG y HTML con los resultados.")

# Función para crear un archivo CSV de ejemplo para pruebas
def crear_archivo_ejemplo(nombre_archivo):
    # Datos de ejemplo que proporcionaste
    datos = """JADA,5704,25.96,92178.61,854.03,1.900,3.820,0.000,-0.001,0.000,0.001,40.952646,-5.618490,1/4/2025,11:49:59
JADA,5705,25.95,92178.09,854.08,1.901,3.820,0.000,-0.001,-0.000,0.001,40.952644,-5.618487,1/4/2025,11:50:0
JADA,5706,25.95,92177.91,854.10,1.900,3.820,0.000,-0.002,-0.000,0.001,40.952646,-5.618485,1/4/2025,11:50:1
JADA,5707,25.95,92177.41,854.14,1.900,3.820,0.000,-0.001,0.000,0.002,40.952646,-5.618473,1/4/2025,11:50:2
JADA,5708,25.95,92177.23,854.16,1.901,3.819,0.000,-0.001,0.000,0.001,40.952647,-5.618467,1/4/2025,11:50:3
JADA,5709,25.95,92177.06,854.17,1.900,3.819,0.000,-0.001,0.000,0.001,40.952647,-5.618460,1/4/2025,11:50:4
JADA,5710,25.96,92177.41,854.14,1.900,3.819,0.000,-0.001,-0.000,0.001,40.952647,-5.618455,1/4/2025,11:50:5
JADA,5711,25.96,92177.08,854.17,1.901,3.819,0.000,-0.001,-0.000,0.001,40.952647,-5.618447,1/4/2025,11:50:6
JADA,5712,25.96,92177.08,854.17,1.901,3.819,0.000,-0.001,-0.000,0.001,40.952649,-5.618431,1/4/2025,11:50:7"""
    
    with open(nombre_archivo, 'w') as f:
        f.write(datos)

if __name__ == "__main__":
    main()