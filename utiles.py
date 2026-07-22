import numpy as np
import cv2

from img_utiles import *

def calcular_matrices_s_direccionales_rgb(imagen_rgb):
    """
    Calcula las magnitudes locales direccionales usadas en la difusión anisotrópica.

    La imagen debe estar en RGB, float32, normalizada en [0, 1].

    Para cada píxel interno se compara el píxel central con sus 4 vecinos cardinales:
        norte, sur, este y oeste.

    Para cada dirección se calcula:
        s = sqrt((delta_R)^2 + (delta_G)^2 + (delta_B)^2)

    Retorna un diccionario con:
        s_norte, s_sur, s_este, s_oeste.
    """
    imagen_rgb = np.clip(imagen_rgb.astype(np.float32), 0.0, 1.0)

    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] not in (1, 3):
        raise ValueError("La imagen debe tener forma (alto, ancho, 1) o (alto, ancho, 3).")

    alto, ancho, _ = imagen_rgb.shape
    if alto < 3 or ancho < 3:
        raise ValueError("La imagen debe tener al menos 3x3 píxeles.")

    centro = imagen_rgb[1:-1, 1:-1, :]

    # Diferencias con vecinos cardinales, igual que en la difusión.
    norte = imagen_rgb[0:-2, 1:-1, :] - centro
    sur = imagen_rgb[2:, 1:-1, :] - centro
    este = imagen_rgb[1:-1, 2:, :] - centro
    oeste = imagen_rgb[1:-1, 0:-2, :] - centro

    # Magnitud de la diferencia RGB.
    s_norte = np.sqrt(np.sum(norte ** 2, axis=2)).astype(np.float32)
    s_sur = np.sqrt(np.sum(sur ** 2, axis=2)).astype(np.float32)
    s_este = np.sqrt(np.sum(este ** 2, axis=2)).astype(np.float32)
    s_oeste = np.sqrt(np.sum(oeste ** 2, axis=2)).astype(np.float32)

    # Cada s_direccion es una matriz con las magnitudes de s para cada direccion
    return {
        "s_norte": s_norte,
        "s_sur": s_sur,
        "s_este": s_este,
        "s_oeste": s_oeste,
    }

def calcular_matriz_s_promedio(s_direccionales, alto, ancho):
    """
    Calcula el promedio de las 4 magnitudes direccionales para cada píxel.

    Recibe el diccionario retornado por calcular_s_direccionales_rgb y las
    dimensiones (alto, ancho) de la imagen original.

    Retorna una matriz float32 de tamaño (alto, ancho), donde cada elemento
    es (s_norte + s_sur + s_este + s_oeste) / 4 para ese píxel.

    Nota:
        Las s_direccionales solo existen para los píxeles internos
        (alto-2, ancho-2). El borde de 1 píxel se rellena replicando
        el valor del píxel interno más cercano.
    """
    s_promedio_interno = (
        s_direccionales["s_norte"]
        + s_direccionales["s_sur"]
        + s_direccionales["s_este"]
        + s_direccionales["s_oeste"]
    ) / 4.0

    # Rellena los bordes
    s_promedio = np.pad(s_promedio_interno, pad_width=1, mode="edge")

    if s_promedio.shape != (alto, ancho):
        raise ValueError("El tamaño resultante no coincide con la imagen.")

    return s_promedio.astype(np.float32)

def calcular_gradiente_medio(imagen):
    """
    Calcula el gradiente medio G sobre la imagen normalizada en [0, 1].

    G es el promedio global de las magnitudes s de diferencia con los
    4 vecinos cardinales (norma RGB): las mismas cantidades que el esquema
    de difusión compara contra lambda en cada iteración.
    """

    # Tener en cuenta que la matriz s_promedio contiene los bordes para que
    # Sea del mismo tamanio de la imagen original
    alto, ancho, _ = imagen.shape

    matrices_s_direccionales = calcular_matrices_s_direccionales_rgb(imagen)

    matriz_s_promedio = calcular_matriz_s_promedio(matrices_s_direccionales, alto, ancho)

    return float(np.mean(matriz_s_promedio))

def calcular_lambda_adaptativo(lambda0, alpha, gradiente_medio):
    """
    Calcula lambda adaptativo usando:

        lambda = lambda0 * (1 + alpha * G)

    donde:
        lambda0 = valor base de lambda
        alpha   = factor de ajuste
        G       = gradiente medio de la imagen ruidosa normalizada
    """
    lambda_adaptativo = lambda0 * (1.0 + alpha * gradiente_medio)

    return float(lambda_adaptativo)

# Deprecado
def calcular_magnitud_gradiente_sobel_deprecado(imagen_rgb):
    """
    Calcula la magnitud del gradiente usando Sobel.

    Parámetros:
        imagen_rgb        : imagen RGB normalizada en [0, 1].
        normalizar_sobel  : si es True, divide la respuesta Sobel por 8.

    La normalización por 8 evita que el gradiente medio G quede en una escala
    demasiado grande al usar una imagen normalizada en [0, 1].
    """
    gris = convertir_a_gris(imagen_rgb)

    escala = 1.0 / 8.0

    grad_x = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3, scale=escala)
    grad_y = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3, scale=escala)

    magnitud = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return magnitud.astype(np.float32)

# Deprecado
def calcular_gradiente_medio_sobel_deprecado(imagen_normalizada):
    """
    Calcula el gradiente medio G sobre la imagen ruidosa normalizada en [0, 1].

    Según la indicación del profesor:
        - G se calcula sobre la imagen ruidosa, antes de iniciar la difusión.
        - La imagen debe estar normalizada en [0, 1].
        - G queda fijo para esa imagen ruidosa durante toda la ejecución.

    Corrección aplicada:
        Sobel se normaliza con scale=1/8 para que G quede en una escala
        coherente con imágenes normalizadas en [0, 1].

    De esta forma, alpha modifica lambda sin dispararlo artificialmente por
    la escala propia del operador Sobel.
    """
    magnitud = calcular_magnitud_gradiente_sobel_deprecado(imagen_normalizada)

    G = np.mean(magnitud)

    return float(G)