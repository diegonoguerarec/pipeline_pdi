import numpy as np

def estimar_divergencia_monte_carlo(
    imagen_ruidosa,
    funcion_caja_negra,
    salida_sin_perturbar,
    epsilon,
    semilla_perturbacion=None
):
    """
    Estima la divergencia de la caja negra por el método Monte Carlo.

    Entrada:
        imagen_ruidosa       : y
        funcion_caja_negra   : función que recibe una imagen y devuelve
                               otra del mismo tamaño (aquí, la difusión)
        salida_sin_perturbar : f(y), ya calculada, para no repetirla
        epsilon              : magnitud de la perturbación
        semilla_perturbacion : entero opcional para reproducir b'.
                               DEBE ser distinta de la semilla usada para
                               generar el ruido. Ver la advertencia abajo.

    Salida:
        divergencia : número real (float), NO dividido por N
    """
    generador = np.random.default_rng(semilla_perturbacion)

    # Vector aleatorio b' de media cero y varianza unidad.
    # Debe tener exactamente la misma forma que la imagen.
    vector_perturbacion = generador.normal(
        loc=0.0,
        scale=1.0,
        size=imagen_ruidosa.shape
    )
    vector_perturbacion = vector_perturbacion.astype(np.float32)

    # Entrada perturbada: y + epsilon * b'
    imagen_perturbada = imagen_ruidosa + epsilon * vector_perturbacion

    # Segunda ejecución de la caja negra.
    salida_perturbada = funcion_caja_negra(imagen_perturbada)

    # Diferencia de salidas: f(y + epsilon*b') - f(y)
    diferencia_salidas = (
        salida_perturbada.astype(np.float64) -
        salida_sin_perturbar.astype(np.float64)
    )

    # Producto interno < b' , diferencia >.
    # Se acumula en float64 porque son cientos de miles de sumandos
    # pequeños y en float32 el error de acumulación sería apreciable.
    producto_interno = np.sum(
        vector_perturbacion.astype(np.float64) * diferencia_salidas
    )

    divergencia = producto_interno / epsilon

    return float(divergencia)


def calcular_SURE(
    imagen_ruidosa,
    funcion_caja_negra,
    sigma,
    epsilon=1e-3,
    semilla_perturbacion=None,
    devolver_terminos=False
):
    """
    Calcula SURE para una caja negra dada, sobre una imagen ruidosa dada.

    CONDICIONES QUE HAY QUE RESPETAR
    --------------------------------
    1. El ruido debe ser gaussiano, aditivo, de media cero, independiente
       píxel a píxel, y con la misma varianza en todos. Si la imagen
       ruidosa fue recortada con np.clip, esto NO se cumple.
    2. sigma debe conocerse o estimarse bien. Un error en sigma se traslada
       directamente al término 3.
    3. La caja negra debe ser continua y débilmente diferenciable. Un
       umbral duro (hard-thresholding) la violaría; una difusión no.

    Entrada:
        imagen_ruidosa     : y, arreglo NumPy, SIN recortar
        funcion_caja_negra : función que recibe una imagen y devuelve otra
                             del mismo tamaño
        sigma              : desviación estándar del ruido (no la varianza)
        epsilon            : magnitud de la perturbación Monte Carlo.
                             El valor por defecto 1e-3 es una decisión de
                             implementación para imágenes normalizadas en
                             [0, 1] en float32: está tres órdenes de
                             magnitud por encima de la resolución de
                             float32 (~1e-7) y tres por debajo del rango
                             dinámico de la imagen. Conviene verificarlo
                             con un barrido en tu caso concreto.
        semilla_perturbacion : entero opcional para reproducir el vector b'.
                             DEBE ser distinta de la semilla del ruido.
        devolver_terminos  : si es True, devuelve además un diccionario
                             con los tres términos por separado

    Salida:
        valor_sure : número real (float)
        (y opcionalmente el diccionario de términos)
    """
    imagen_ruidosa = imagen_ruidosa.astype(np.float32)

    numero_de_pixeles = imagen_ruidosa.size

    # ---- Ejecución 1 de la caja negra: f(y) ----
    salida_sin_perturbar = funcion_caja_negra(imagen_ruidosa)

    if salida_sin_perturbar.shape != imagen_ruidosa.shape:
        raise ValueError(
            "La caja negra debe devolver una imagen del mismo tamaño "
            "que la de entrada."
        )

    # ---- Término 1: (1/N) * || f(y) - y ||^2 ----
    residuo = (
        salida_sin_perturbar.astype(np.float64) -
        imagen_ruidosa.astype(np.float64)
    )

    termino_ajuste_a_los_datos = np.sum(residuo ** 2) / numero_de_pixeles

    # ---- Término 2: - sigma^2 ----
    termino_constante = -(sigma ** 2)

    # ---- Término 3: (2*sigma^2/N) * div ----
    # Ejecución 2 de la caja negra, dentro de esta función.
    divergencia = estimar_divergencia_monte_carlo(
        imagen_ruidosa=imagen_ruidosa,
        funcion_caja_negra=funcion_caja_negra,
        salida_sin_perturbar=salida_sin_perturbar,
        epsilon=epsilon,
        semilla_perturbacion=semilla_perturbacion
    )

    termino_divergencia = (
        2.0 * (sigma ** 2) * divergencia / numero_de_pixeles
    )

    # ---- Suma final ----
    valor_sure = (
        termino_ajuste_a_los_datos +
        termino_constante +
        termino_divergencia
    )

    if devolver_terminos:
        terminos = {
            "termino_ajuste_a_los_datos": float(termino_ajuste_a_los_datos),
            "termino_constante": float(termino_constante),
            "termino_divergencia": float(termino_divergencia),
            "divergencia": float(divergencia),
            "divergencia_normalizada": float(divergencia / numero_de_pixeles),
            "numero_de_pixeles": int(numero_de_pixeles)
        }
        return float(valor_sure), terminos

    return float(valor_sure)


def calcular_mse_real(imagen_limpia, imagen_procesada):
    """
    Calcula el MSE verdadero: (1/N) * || f(y) - x ||^2

    Esta función NO forma parte de SURE. Existe solo para validarlo.

    Esa comparación es la única forma seria de comprobar que SURE funciona
    antes de usarlo a ciegas.

    Entrada:
        imagen_limpia    : x
        imagen_procesada : f(y)

    Salida:
        mse : número real (float)
    """
    imagen_limpia = imagen_limpia.astype(np.float64)
    imagen_procesada = imagen_procesada.astype(np.float64)

    if imagen_limpia.shape != imagen_procesada.shape:
        raise ValueError(
            "Las imágenes deben tener la misma dimensión para calcular MSE."
        )

    diferencia = imagen_procesada - imagen_limpia

    mse = np.sum(diferencia ** 2) / imagen_limpia.size

    return float(mse)