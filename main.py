from img_utiles import *
from difusion import *

if __name__ == '__main__':
    imagen = leer_imagen_ppm_rgb(ruta_imagen='entrada/im0001.ppm')

    # Usar solo canal verde
    imagen = imagen[:, :, 1:2]

    imagen_difundida = difusion_anisotropica_rgb(imagen=imagen)

    guardar_imagen_rgb(imagen_rgb=imagen, ruta_salida='salida/im0001.png')
    guardar_imagen_rgb(imagen_rgb=imagen_difundida, ruta_salida='salida/im0001_difundida.png')