import os
import numpy as np
import cv2
import scipy.io

# Ruta del dataset
ruta_dataset = r"D:\ia1-repositorio\PRACTICO-2\dataset_raw"

# Inicializar listas para almacenar imágenes y etiquetas
imagenes = []
etiquetas = []

# Recorrer todas las carpetas dentro de dataset_raw
for etiqueta in range(10):  # Asumiendo que las carpetas son dig0, dig1, ..., dig9
    carpeta = os.path.join(ruta_dataset, f"dig{etiqueta}")
    
    # Obtener todas las imágenes de la carpeta y ordenarlas
    archivos = sorted(os.listdir(carpeta))  # Asegurar orden correcto
    
    for archivo in archivos:
        ruta_imagen = os.path.join(carpeta, archivo)
        
        # Leer la imagen en escala de grises y normalizar
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # Asegurar que la imagen es de 20x20
        if imagen.shape != (20, 20):
            print(f"Advertencia: {archivo} tiene dimensiones {imagen.shape}, debe ser 20x20")
            continue
        
        # Aplanar la imagen de 20x20 a un vector de 400 elementos
        imagenes.append(imagen.flatten())
        etiquetas.append(etiqueta)

# Convertir a arrays de NumPy
imagenes = np.array(imagenes, dtype=np.float32)
etiquetas = np.array(etiquetas, dtype=np.int32)

# Unir imágenes y etiquetas en un solo dataset
dataset = np.column_stack((imagenes, etiquetas))

# Guardar en .npy
np.save("D:\ia1-repositorio\PRACTICO-2\dataset_mnist/dataset.npy", dataset)
print("Dataset guardado en dataset.npy")

# Guardar en .mat
scipy.io.savemat("D:\ia1-repositorio\PRACTICO-2\dataset_mnist/dataset.mat", {"dataset": dataset})
print("Dataset guardado en dataset.mat")
