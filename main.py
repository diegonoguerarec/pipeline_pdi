from img_utiles import *
from ruidos import *
from difusion import *
from utiles import *
from metricas import *

if __name__ == '__main__':
    # Seteo de variables

    # Carpeta que contiene las imágenes en formato .ppm
    carpeta_entrada = 'entrada'

    carpeta_salida = 'salida'

    # 6 niveles tomados como referencia del paper para ruido Gaussiano.
    # Para Speckle usamos los mismos valores para poder comparar niveles.
    niveles_ruido = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

    # Para Poisson usamos 6 niveles propios.
    # En Poisson no usamos varianza directamente.
    # Menor pico = más ruido.
    picos_poisson = [255, 150, 100, 75, 50, 30]

    # Lambda adaptativo:
    # lambda = lambda0 * (1 + alpha * G)
    # donde G es el gradiente medio de la imagen ruidosa normalizada en [0, 1].
    # En esta versión, G se calcula hallando manualmente las diferencias de cada
    # pixel con sus 4 vecinos, promediandolo para cada pixel y luego sacando 
    # el promedio de la matriz de gradientes locales
    lambda0 = 0.1
    alphas = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100]