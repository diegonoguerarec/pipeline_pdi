import cv2
import numpy as np

from img_utiles import *

def calcular_psnr(imagen_original, imagen_procesada, data_range=1.0):
    """
    Calcula PSNR entre la imagen original y la imagen procesada.

    Las imágenes se trabajan normalizadas en [0, 1], por eso data_range=1.0.

    Para imágenes RGB, se calcula el PSNR por canal y luego se promedia,
    siguiendo el criterio del paper: las métricas se computan por canal RGB
    y se promedian para obtener un único valor por imagen.

    Mayor PSNR => menor diferencia respecto a la original.
    """
    imagen_original = np.clip(imagen_original.astype(np.float32), 0.0, data_range)
    imagen_procesada = np.clip(imagen_procesada.astype(np.float32), 0.0, data_range)

    if imagen_original.shape != imagen_procesada.shape:
        raise ValueError("Las imágenes deben tener la misma dimensión para calcular PSNR.")

    # Si la imagen tiene 3 canales, calculamos PSNR por canal y promediamos.
    if imagen_original.ndim == 3:
        valores_psnr = []

        for canal in range(imagen_original.shape[2]):
            mse = np.mean(
                (imagen_original[:, :, canal] - imagen_procesada[:, :, canal]) ** 2
            )

            if mse == 0:
                valores_psnr.append(float("inf"))
            else:
                valores_psnr.append(10.0 * np.log10((data_range ** 2) / mse))

        return float(np.mean(valores_psnr))

    mse = np.mean((imagen_original - imagen_procesada) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10.0 * np.log10((data_range ** 2) / mse)

    return float(psnr)

# Se usa para calcular EPI
def _correlacion_pearson(x, y):
    """
    Calcula correlación de Pearson entre dos arreglos.
    """
    x = x.astype(np.float32).ravel()
    y = y.astype(np.float32).ravel()

    x_media = np.mean(x)
    y_media = np.mean(y)

    x_centrada = x - x_media
    y_centrada = y - y_media

    numerador = np.sum(x_centrada * y_centrada)
    denominador = np.sqrt(
        np.sum(x_centrada ** 2) *
        np.sum(y_centrada ** 2)
    )

    if denominador <= 1e-12:
        return 0.0

    return float(numerador / denominador)

def calcular_epi(imagen_original, imagen_procesada):
    """
    Calcula EPI, Edge Preservation Index, usando la definición del paper.

    El paper calcula EPI como la correlación de Pearson entre los Laplacianos
    de la imagen original y la imagen procesada.

    Kernel Laplaciano usado:
        [[1/4, 1/2, 1/4],
         [1/2, -3 , 1/2],
         [1/4, 1/2, 1/4]]

    Para imágenes RGB:
        1. Se calcula el Laplaciano por canal.
        2. Se calcula la correlación por canal.
        3. Se promedian las correlaciones de R, G y B.

    EPI cercano a 1 => mejor preservación de bordes.
    """
    imagen_original = np.clip(imagen_original.astype(np.float32), 0.0, 1.0)
    imagen_procesada = np.clip(imagen_procesada.astype(np.float32), 0.0, 1.0)

    if imagen_original.shape != imagen_procesada.shape:
        raise ValueError("Las imágenes deben tener la misma dimensión para calcular EPI.")

    kernel_laplaciano = np.array(
        [
            [1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0],
            [1.0 / 2.0, -3.0,      1.0 / 2.0],
            [1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0]
        ],
        dtype=np.float32
    )

    # Caso RGB: calcular por canal y promediar.
    if imagen_original.ndim == 3:
        valores_epi = []

        for canal in range(imagen_original.shape[2]):
            lap_original = cv2.filter2D(
                imagen_original[:, :, canal],
                cv2.CV_32F,
                kernel_laplaciano,
                borderType=cv2.BORDER_REFLECT
            )
            lap_procesada = cv2.filter2D(
                imagen_procesada[:, :, canal],
                cv2.CV_32F,
                kernel_laplaciano,
                borderType=cv2.BORDER_REFLECT
            )

            valores_epi.append(_correlacion_pearson(lap_original, lap_procesada))

        return float(np.mean(valores_epi))

    lap_original = cv2.filter2D(
        imagen_original,
        cv2.CV_32F,
        kernel_laplaciano,
        borderType=cv2.BORDER_REFLECT
    )
    lap_procesada = cv2.filter2D(
        imagen_procesada,
        cv2.CV_32F,
        kernel_laplaciano,
        borderType=cv2.BORDER_REFLECT
    )

    return _correlacion_pearson(lap_original, lap_procesada)

def calcular_ssim_rgb(imagen_original, imagen_procesada, data_range=1.0):
    """
    Calcula SSIM para una imagen RGB.

    Implementación manual basada en la fórmula clásica de SSIM.

    Se calcula SSIM por canal:
        R, G y B

    Luego se promedia el resultado.
    """
    imagen_original = imagen_original.astype(np.float32)
    imagen_procesada = imagen_procesada.astype(np.float32)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    valores_ssim = []

    for canal in range(imagen_original.shape[2]):
        x = imagen_original[:, :, canal]
        y = imagen_procesada[:, :, canal]

        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x ** 2, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y ** 2, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

        mapa_ssim = (
            ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
            ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        )

        valores_ssim.append(np.mean(mapa_ssim))

    ssim_promedio = np.mean(valores_ssim)

    return float(ssim_promedio)

# ------------------------------------------------------------
# FSIM: Feature Similarity Index
# ------------------------------------------------------------
# Queda revisar si se puede reemplazar el calculo de la magnitud del gradiente
# por algun metodo similar al que se usa en el resto del código, evitando usar SOBEL
def _calcular_magnitud_gradiente_gris(imagen_gris):
    """
    Calcula la magnitud del gradiente de una imagen en escala de grises.

    FSIM usa la magnitud del gradiente (GM) como característica secundaria,
    porque el gradiente representa información de contraste y bordes.
    """
    imagen_gris = imagen_gris.astype(np.float32)

    grad_x = cv2.Sobel(imagen_gris, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagen_gris, cv2.CV_32F, 0, 1, ksize=3)

    magnitud = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return magnitud.astype(np.float32)

def calcular_phase_congruency_aproximada(
    imagen_gris,
    nscale=3,
    norient=4,
    min_wavelength=6.0,
    mult=2.0,
    sigma_on_f=0.55
):
    """
    Calcula una aproximación práctica del mapa de Phase Congruency (PC).

    En el paper de FSIM, PC se usa como característica principal porque
    representa la importancia de la estructura local de la imagen.

    La implementación completa de Kovesi incluye varios pasos adicionales
    de compensación de ruido y ponderación. Para mantener el código dentro
    de NumPy/OpenCV y que sea entendible, aquí usamos una versión práctica:

        1. Convertimos la imagen gris al dominio de Fourier.
        2. Aplicamos filtros log-Gabor en varias escalas y orientaciones.
        3. Calculamos energía local y amplitud local.
        4. Obtenemos PC como energía / amplitud.

    Entrada:
        imagen_gris: imagen en escala de grises normalizada en [0, 1].

    Salida:
        pc: mapa de phase congruency en rango aproximado [0, 1].
    """
    imagen_gris = np.clip(imagen_gris.astype(np.float32), 0.0, 1.0)

    filas, columnas = imagen_gris.shape
    epsilon = 1e-8

    # Coordenadas de frecuencia sin desplazar, compatibles con np.fft.fft2.
    freq_y = np.fft.fftfreq(filas)
    freq_x = np.fft.fftfreq(columnas)
    x, y = np.meshgrid(freq_x, freq_y)

    radio = np.sqrt(x ** 2 + y ** 2)
    angulo = np.arctan2(-y, x)

    # Evita log(0) en la frecuencia cero.
    radio[0, 0] = 1.0

    imagen_fft = np.fft.fft2(imagen_gris)

    pc_total = np.zeros_like(imagen_gris, dtype=np.float32)

    # Ancho angular de los filtros. Controla cuánto cubre cada orientación.
    theta_sigma = (np.pi / norient) / 1.5

    for o in range(norient):
        angulo_orientacion = o * np.pi / norient

        # Diferencia angular envuelta en [-pi, pi].
        diferencia_angular = np.arctan2(
            np.sin(angulo - angulo_orientacion),
            np.cos(angulo - angulo_orientacion)
        )

        filtro_angular = np.exp(
            -(diferencia_angular ** 2) / (2 * theta_sigma ** 2)
        )

        suma_par = np.zeros_like(imagen_gris, dtype=np.float32)
        suma_impar = np.zeros_like(imagen_gris, dtype=np.float32)
        suma_amplitud = np.zeros_like(imagen_gris, dtype=np.float32)

        for s in range(nscale):
            longitud_onda = min_wavelength * (mult ** s)
            frecuencia_central = 1.0 / longitud_onda

            filtro_radial = np.exp(
                -(np.log(radio / frecuencia_central) ** 2) /
                (2 * (np.log(sigma_on_f) ** 2))
            )
            filtro_radial[0, 0] = 0.0

            filtro_log_gabor = filtro_radial * filtro_angular

            respuesta = np.fft.ifft2(imagen_fft * filtro_log_gabor)

            respuesta_par = np.real(respuesta).astype(np.float32)
            respuesta_impar = np.imag(respuesta).astype(np.float32)
            amplitud = np.sqrt(respuesta_par ** 2 + respuesta_impar ** 2)

            suma_par += respuesta_par
            suma_impar += respuesta_impar
            suma_amplitud += amplitud

        energia = np.sqrt(suma_par ** 2 + suma_impar ** 2)
        pc_orientacion = energia / (suma_amplitud + epsilon)

        pc_total += pc_orientacion.astype(np.float32)

    pc = pc_total / float(norient)
    pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
    pc = np.clip(pc, 0.0, 1.0)

    return pc.astype(np.float32)

def calcular_caracteristicas_fsim(imagen_rgb):
    """
    Obtiene las dos características usadas por FSIM:

        PC: Phase Congruency, característica principal.
        GM: Gradient Magnitude, característica secundaria.

    El paper indica que FSIM fue diseñado para imágenes en escala de grises
    o para el canal de luminancia de imágenes a color. Por eso aquí primero
    convertimos la imagen RGB a gris/luminancia.
    """
    #gris = convertir_a_gris(imagen_rgb)

    # Pipeline en canal verde: la "imagen" ya es un solo canal, se usa
    # directamente como entrada a PC y GM (sin conversión a gris ponderado).
    gris = imagen_rgb[:, :, 0]

    pc = calcular_phase_congruency_aproximada(gris)
    gm = _calcular_magnitud_gradiente_gris(gris)

    return pc, gm

def calcular_fsim(
    imagen_original,
    imagen_procesada,
    caracteristicas_original=None,
    T1=0.85,
    T2=None
):
    """
    Calcula FSIM entre una imagen original y una imagen evaluada.

    FSIM compara dos tipos de características:

        1. PC: Phase Congruency.
        2. GM: Gradient Magnitude.

    Luego construye un mapa de similitud local y lo promedia usando PC
    como peso, porque las zonas con más estructura visual tienen mayor
    importancia perceptual.

    Como nuestras imágenes están normalizadas en [0, 1], usamos T2 escalado.
    El paper usa T2 = 160 para imágenes con intensidades en [0, 255].
    Al trabajar en [0, 1], se usa:

        T2 = 160 / (255^2)

    Interpretación:
        FSIM cercano a 1  -> imágenes muy similares.
        FSIM cercano a 0  -> imágenes poco similares.
    """
    if T2 is None:
        T2 = 160.0 / (255.0 ** 2)

    if caracteristicas_original is None:
        pc_original, gm_original = calcular_caracteristicas_fsim(imagen_original)
    else:
        pc_original, gm_original = caracteristicas_original

    pc_procesada, gm_procesada = calcular_caracteristicas_fsim(imagen_procesada)

    pc_original = pc_original.astype(np.float32)
    pc_procesada = pc_procesada.astype(np.float32)
    gm_original = gm_original.astype(np.float32)
    gm_procesada = gm_procesada.astype(np.float32)

    similitud_pc = (
        (2.0 * pc_original * pc_procesada + T1) /
        (pc_original ** 2 + pc_procesada ** 2 + T1)
    )

    similitud_gm = (
        (2.0 * gm_original * gm_procesada + T2) /
        (gm_original ** 2 + gm_procesada ** 2 + T2)
    )

    # En el paper se usan exponentes alpha = beta = 1 para combinar PC y GM.
    similitud_local = similitud_pc * similitud_gm

    # Peso visual: se da más importancia a zonas donde alguna de las dos
    # imágenes tiene mayor phase congruency.
    peso = np.maximum(pc_original, pc_procesada)

    denominador = np.sum(peso)

    if denominador <= 1e-12:
        return float(np.mean(similitud_local))

    fsim = np.sum(similitud_local * peso) / denominador

    return float(fsim)

