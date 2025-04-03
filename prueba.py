import pandas as pd
import matplotlib.pyplot as plt
import simplekml

def cargar_datos(archivo_csv):
    columnas = ["ID", "Tiempo", "Temperatura", "Presion", "Altitud", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "Latitud", "Longitud", "Fecha", "Hora"]
    df = pd.read_csv(archivo_csv, names=columnas)
    return df

def graficar_datos(df):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 1, 1)
    plt.plot(df["Tiempo"], df["Altitud"], label="Altitud (m)", color='blue')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Altitud (m)")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(df["Tiempo"], df["Temperatura"], label="Temperatura (°C)", color='red')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def generar_kml(df, nombre_archivo="trayectoria.kml"):
    kml = simplekml.Kml()
    
    for _, fila in df.iterrows():
        if fila["Latitud"] != "INVALID" and fila["Longitud"] != "INVALID":
            kml.newpoint(name=f"Tiempo {fila['Tiempo']}", coords=[(float(fila["Longitud"]), float(fila["Latitud"]))])
    
    kml.save(nombre_archivo)
    print(f"Archivo KML generado: {nombre_archivo}")

# --- Programa principal ---
archivo_csv = "datos.csv"  # Nombre del archivo CSV con los datos

df = cargar_datos(archivo_csv)
graficar_datos(df)
generar_kml(df)
