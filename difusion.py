import numpy as np

def difusividad_perona_malik(s, lambda_):
    """
    Función de difusividad de Perona-Malik:

        g(s) = exp(-(s^2 / lambda^2))

    Donde:
        s       = magnitud de diferencia local
        lambda  = parámetro de sensibilidad a bordes

    Si s es pequeño:
        g(s) se acerca a 1, entonces se permite suavizar.

    Si s es grande:
        g(s) se acerca a 0, entonces se frena la difusión.
    """
    return np.exp(-(s ** 2) / (lambda_ ** 2))

def difusion_anisotropica_rgb(imagen, lambda_=0.1, delta_t=0.15, iteraciones=35):
    """
    Aplica difusión anisotrópica a una imagen RGB.

    Entrada:
        imagen      : imagen RGB normalizada en [0, 1]
        lambda_     : parámetro de difusividad
        delta_t     : paso temporal
        iteraciones : cantidad de iteraciones

    Salida:
        imagen procesada con difusión anisotrópica en [0, 1]

    En este experimento NO se aplican filtros clásicos antes.
    La difusión se aplica directamente sobre la imagen ruidosa.
    """
    u = imagen.astype(np.float32).copy()

    for _ in range(iteraciones):
        u_anterior = u.copy()

        centro = u_anterior[1:-1, 1:-1, :]

        # Diferencias con vecinos cardinales
        norte = u_anterior[0:-2, 1:-1, :] - centro
        sur = u_anterior[2:, 1:-1, :] - centro
        este = u_anterior[1:-1, 2:, :] - centro
        oeste = u_anterior[1:-1, 0:-2, :] - centro

        # Magnitud de diferencia para imagen RGB.
        # Se combinan los tres canales.
        s_norte = np.sqrt(np.sum(norte ** 2, axis=2))
        s_sur = np.sqrt(np.sum(sur ** 2, axis=2))
        s_este = np.sqrt(np.sum(este ** 2, axis=2))
        s_oeste = np.sqrt(np.sum(oeste ** 2, axis=2))

        # Coeficientes de difusión
        g_norte = difusividad_perona_malik(s_norte, lambda_)
        g_sur = difusividad_perona_malik(s_sur, lambda_)
        g_este = difusividad_perona_malik(s_este, lambda_)
        g_oeste = difusividad_perona_malik(s_oeste, lambda_)

        # Expandimos dimensiones para poder multiplicar por los 3 canales RGB
        g_norte = g_norte[:, :, np.newaxis]
        g_sur = g_sur[:, :, np.newaxis]
        g_este = g_este[:, :, np.newaxis]
        g_oeste = g_oeste[:, :, np.newaxis]

        # Flujo total de difusión
        flujo = (
            g_norte * norte +
            g_sur * sur +
            g_este * este +
            g_oeste * oeste
        )

        # Esquema explícito de Euler
        u[1:-1, 1:-1, :] = centro + delta_t * flujo

        # Mantenemos valores válidos
        u = np.clip(u, 0.0, 1.0)

    return u