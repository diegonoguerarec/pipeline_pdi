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
    ##imagen_ruidosa = np.clip(imagen_ruidosa, 0.0, 1.0)

    return imagen_ruidosa