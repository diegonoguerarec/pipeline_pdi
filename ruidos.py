import numpy as np

def agregar_ruido_gaussiano(imagen, varianza=0.01, seed=None):
    """
    Agrega ruido Gaussiano aditivo.

    Fórmula:
        imagen_ruidosa = imagen + ruido

    donde:
        ruido ~ N(0, varianza)

    La imagen debe estar normalizada en [0, 1].
    """
    rng = np.random.default_rng(seed)

    media = 0.0
    desviacion = np.sqrt(varianza)

    ruido = rng.normal(
        loc=media,
        scale=desviacion,
        size=imagen.shape
    ).astype(np.float32)

    imagen_ruidosa = imagen + ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0.0, 1.0)

    return imagen_ruidosa


def agregar_ruido_speckle(imagen, varianza=0.01, seed=None):
    """
    Agrega ruido Speckle.

    El ruido Speckle es multiplicativo.

    Fórmula:
        imagen_ruidosa = imagen + imagen * ruido

    donde:
        ruido ~ N(0, varianza)

    La imagen debe estar normalizada en [0, 1].
    """
    rng = np.random.default_rng(seed)

    media = 0.0
    desviacion = np.sqrt(varianza)

    ruido = rng.normal(
        loc=media,
        scale=desviacion,
        size=imagen.shape
    ).astype(np.float32)

    imagen_ruidosa = imagen + imagen * ruido
    imagen_ruidosa = np.clip(imagen_ruidosa, 0.0, 1.0)

    return imagen_ruidosa


def agregar_ruido_poisson(imagen, pico=255, seed=None):
    """
    Agrega ruido Poisson.

    Este ruido se modela como un conteo de fotones.

    Idea:
        conteos = Poisson(imagen * pico)
        imagen_ruidosa = conteos / pico

    Importante:
        pico mayor  -> menos ruido
        pico menor  -> más ruido
    """
    rng = np.random.default_rng(seed)

    imagen_escalada = imagen * pico
    imagen_ruidosa = rng.poisson(imagen_escalada).astype(np.float32) / pico

    imagen_ruidosa = np.clip(imagen_ruidosa, 0.0, 1.0)

    return imagen_ruidosa