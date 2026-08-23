import cv2
import numpy as np
from pathlib import Path

def leer_imagen_ppm_rgb(ruta_imagen):   
    """
    Lee una imagen .ppm usando OpenCV.

    OpenCV carga imágenes a color en formato BGR.
    Para trabajar de manera más clara, convertimos a RGB.

    Retorna:
        imagen_rgb: imagen en formato RGB, tipo float32, rango [0, 1].
    """
    imagen_bgr = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)

    if imagen_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_rgb = imagen_rgb.astype(np.float32) / 255.0

    return imagen_rgb

def guardar_imagen_rgb(ruta_salida, imagen_rgb):
    """
    Guarda una imagen RGB normalizada en [0, 1] como PNG.

    Como OpenCV guarda imágenes en BGR, antes de guardar
    convertimos de RGB a BGR.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    imagen_rgb = np.clip(imagen_rgb, 0.0, 1.0)
    imagen_uint8 = (imagen_rgb * 255).round().astype(np.uint8)

    imagen_bgr = cv2.cvtColor(imagen_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(ruta_salida), imagen_bgr)