import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Nombre del archivo CSV
def process_cansat_data(file_path):
    # Definir nombres de columnas según el formato de los datos
    columns = ["Tag", "Index", "Temperature", "Pressure", "Altitude", 
               "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", 
               "GPS_Lat", "GPS_Lon", "Date", "Time"]
    
    # Cargar datos
    df = pd.read_csv(file_path, names=columns, delimiter=",", skipinitialspace=True)
    
    # Convertir tipos de datos
    df["Index"] = df["Index"].astype(int)
    df["Temperature"] = df["Temperature"].astype(float)
    df["Pressure"] = df["Pressure"].astype(float)
    df["Altitude"] = df["Altitude"].astype(float)
    df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
    
    # Graficar Altitud vs. Index
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["Index"], y=df["Altitude"], marker="o", label="Altitud (m)")
    plt.xlabel("Tiempo (Índice)")
    plt.ylabel("Altitud (m)")
    plt.title("Altitud vs. Tiempo")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Graficar Presión vs. Altitud
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df["Altitude"], y=df["Pressure"], marker="o")
    plt.xlabel("Altitud (m)")
    plt.ylabel("Presión (Pa)")
    plt.title("Presión vs. Altitud")
    plt.grid()
    plt.show()
    
    # Graficar Aceleraciones
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["Index"], y=df["AccX"], label="AccX")
    sns.lineplot(x=df["Index"], y=df["AccY"], label="AccY")
    sns.lineplot(x=df["Index"], y=df["AccZ"], label="AccZ")
    plt.xlabel("Tiempo (Índice)")
    plt.ylabel("Aceleración (g)")
    plt.title("Aceleración en cada eje vs. Tiempo")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Graficar Temperatura vs. Tiempo
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["Index"], y=df["Temperature"], marker="o", label="Temperatura (°C)")
    plt.xlabel("Tiempo (Índice)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Temperatura vs. Tiempo")
    plt.legend()
    plt.grid()
    plt.show()

def crear_kml(datos_gps, archivo_kml):
    # Cabecera del archivo KML
    kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
    <name>Ruta CanSat</name>
    <description>Recorrido del CanSat en el vuelo</description>
    '''

    # Iterar sobre los datos GPS y agregar cada punto al KML
    for row in datos_gps:
        lat = row[2]  # Latitud (ajustar según formato CSV)
        lon = row[3]  # Longitud
        alt = row[4]  # Altitud
        timestamp = row[1]  # Número de paquete o timestamp (opcional)

        kml_content += f'''
        <Placemark>
            <name>{timestamp}</name>
            <description>Altitud: {alt} m</description>
            <Point>
                <coordinates>{lon},{lat},{alt}</coordinates>
            </Point>
        </Placemark>
        '''

    # Final del archivo KML
    kml_content += '''
    </Document>
    </kml>
    '''

    # Guardar el archivo KML
    with open(archivo_kml, 'w') as file:
        file.write(kml_content)

# Leer los datos del GPS desde un archivo CSV
def leer_datos_csv(archivo_csv):
    datos_gps = []
    with open(archivo_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] != "JADA":  # Asegurarse de que no se procesa la cabecera
                datos_gps.append(row)
    return datos_gps

# Ruta del archivo CSV con los datos GPS
archivo_csv = "datos.csv"

# Leer los datos GPS del archivo CSV
datos_gps = leer_datos_csv(archivo_csv)

# Generar el archivo KML
crear_kml(datos_gps, 'ruta_cansat.kml')
print("¡Archivo KML generado con éxito! Puedes abrirlo en Google Earth.")

# Llamada al programa con el archivo CSV
data_file = "datos.csv"  # Cambia esto al nombre real del archivo
process_cansat_data(data_file)
