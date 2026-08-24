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

    POR QUÉ HACE FALTA ESTIMARLA
    ----------------------------
    La divergencia es:

            div = sum_n  d f_n(y) / d y_n

    Es decir: cuánto cambia el píxel n de la SALIDA cuando movemos un
    poquito el píxel n de la ENTRADA, sumado sobre todos los píxeles.
    Eso es la traza de la matriz Jacobiana de f.

    Calcularla exactamente exigiría N ejecuciones de la caja negra (una
    por píxel, perturbando un píxel a la vez). Con una imagen de 605x700
    eso son 423.500 ejecuciones de la difusión. Inviable.

    LA IDEA DE RAMANI
    -----------------
    En lugar de perturbar un píxel a la vez, se perturban TODOS a la vez
    con un vector aleatorio b' de media cero y varianza unidad, y se
    proyecta el resultado de vuelta sobre b':

            div  ~=  (1/epsilon) * < b' , f(y + epsilon*b') - f(y) >

    donde <a, c> es el producto interno (suma de a_i * c_i).

    POR QUÉ FUNCIONA
    ----------------
    Por desarrollo de Taylor de primer orden:

            f(y + epsilon*b')  ~=  f(y) + epsilon * J * b'

    donde J es la matriz Jacobiana. Restando f(y) y dividiendo por epsilon:

            (1/epsilon) * [ f(y + epsilon*b') - f(y) ]  ~=  J * b'

    Multiplicando por b' por la izquierda:

            b'^T * J * b'  =  sum_i sum_j  b'_i * J_ij * b'_j

    Ahora tomamos el valor esperado sobre b'. Como las componentes de b'
    son independientes, de media cero y varianza uno:

            E[ b'_i * b'_j ]  =  1  si i = j
            E[ b'_i * b'_j ]  =  0  si i != j

    Entonces todos los términos cruzados (i != j) se cancelan en promedio
    y sobrevive únicamente la diagonal:

            E[ b'^T * J * b' ]  =  sum_i J_ii  =  traza(J)  =  div

    Que es exactamente lo que buscábamos. Con una sola realización de b'
    basta, porque la suma sobre N píxeles ya promedia por sí sola.

    COSTE
    -----
    Una ejecución extra de la caja negra. Nada más.

    ADVERTENCIA: b' DEBE SER INDEPENDIENTE DEL RUIDO b
    --------------------------------------------------
    La demostración de arriba usa que b' es independiente de todo lo demás.
    Si b' resulta estar correlacionado con el ruido b que ya está dentro
    de y, la estimación se corrompe.

    El error es fácil de cometer sin darse cuenta: basta con pasar la misma
    semilla al generador de ruido y a esta función. Como ambos usan
    np.random.default_rng, la misma semilla produce exactamente la misma
    secuencia, y entonces b' = b / sigma, es decir, correlación 1.0.

    Medido sobre im0001.ppm (sigma=0.05, lambda=0.075, MSE real 2.7969e-4):

        misma semilla (corr(b, b') = 1.000000):
            SURE = 6.4643e-4    error +3.67e-4   (130% de error)

        12 semillas independientes (corr(b, b') ~ 0.0001):
            SURE = 2.8687e-4 de media, desviación 8.2e-6
            error medio +7.2e-6   (2.56% de error)

    Por eso los parámetros se llaman "semilla_ruido" y
    "semilla_perturbacion": para que la colisión sea visible al leer el
    código que las llama. Si no necesitás reproducibilidad, dejá
    semilla_perturbacion=None y el problema no puede ocurrir.

    SOBRE epsilon
    -------------
    Hay dos presiones opuestas:
        - epsilon debe ser PEQUEÑO para que el desarrollo de Taylor de
          primer orden sea válido (para imitar el límite epsilon -> 0).
        - epsilon debe ser GRANDE para que la diferencia
          f(y + epsilon*b') - f(y) no se pierda en el error de redondeo
          de la aritmética de punto flotante.

    Ramani et al. reportan que el rango admisible cubre varias décadas,
    de modo que la elección no es crítica, siempre que se esté dentro de
    ese rango. La manera correcta de fijarlo es empírica: barrer epsilon
    y quedarse en la meseta donde el valor estimado no cambia.

    Meseta medida sobre im0001.ppm (canal verde, 605x700, sigma=0.05,
    lambda=0.075, delta_t=0.15, 35 iteraciones), comparando contra el
    MSE verdadero:

        epsilon    div/N      error de SURE frente al MSE real
        -------    -------    --------------------------------
        1e-8       0.01903    -3.7e-5   <- ya degradado por redondeo
        1e-7       0.02694    +2.7e-6
        1e-6       0.02692    +2.6e-6
        1e-5       0.02692    +2.6e-6
        1e-4       0.02748    +5.4e-6
        1e-3       0.02791    +7.6e-6
        1e-2       0.02815    +8.8e-6
        1e-1       0.35318    +1.6e-3   <- Taylor de 1er orden ya no vale
        5e-1       0.61082    +2.9e-3

    La meseta abarca de 1e-7 a 1e-2, unas cinco décadas, coincidiendo con
    lo que reporta el artículo. Por debajo el redondeo de float32 destruye
    la diferencia; por encima la aproximación lineal deja de ser válida y
    la divergencia se sobreestima de forma masiva.

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

    LA FÓRMULA, TÉRMINO POR TÉRMINO
    -------------------------------
        SURE = (1/N)*||f(y) - y||^2  -  sigma^2  +  (2*sigma^2/N)*div

    Término 1: (1/N) * || f(y) - y ||^2
        Distancia media al cuadrado entre la salida y la ENTRADA RUIDOSA.
        Es lo único que podemos medir directamente, porque no involucra x.
        Si el filtro no hace nada (f(y) = y), este término vale 0.
        Cuanto más suaviza el filtro, más crece.

    Término 2: - sigma^2
        Corrección constante. Sale de la descomposición del MSE: el término
        (1/N)*||y - x||^2 vale sigma^2 en promedio, y aparece con signo
        cambiado al reordenar. No depende de los parámetros del filtro,
        así que no afecta a dónde está el mínimo, pero sí al valor absoluto.

    Término 3: (2*sigma^2/N) * div
        El término de penalización. Mide cuánto sigue la salida al ruido
        de la entrada. Si el filtro copia la entrada tal cual, cada
        derivada d f_n / d y_n vale 1, la divergencia vale N, y este
        término vale 2*sigma^2: la penalización máxima. Si el filtro
        aplasta todo a una constante, las derivadas valen 0, la divergencia
        vale 0, y no hay penalización.

    Los términos 1 y 3 tiran en direcciones opuestas, y ese es todo el
    mecanismo de SURE:
        - Suavizar poco  -> término 1 pequeño, término 3 grande
                            (el filtro está reproduciendo el ruido)
        - Suavizar mucho -> término 1 grande, término 3 pequeño
                            (el filtro está borrando la señal)
    El mínimo de la suma es el punto de equilibrio, y ese es el punto que
    aproxima el mínimo del MSE verdadero.

    DE DÓNDE SALE LA FÓRMULA
    ------------------------
    Partimos del MSE y sumamos y restamos y dentro de la norma:

        (1/N)*||f(y) - x||^2 = (1/N)*||f(y) - y + y - x||^2

    Desarrollando el cuadrado de la suma, y usando que y - x = b:

        = (1/N)*||f(y) - y||^2
          + (2/N)*< f(y) - y , b >
          + (1/N)*||b||^2

    Ahora tomamos el valor esperado de cada pieza:

        - E[ (1/N)*||b||^2 ]  =  sigma^2, por definición de la varianza.

        - (2/N)*< f(y) - y , b >  se parte en dos:
              (2/N)*< f(y), b >  -  (2/N)*< y - x , b >
          El segundo trozo es (2/N)*||b||^2, cuya esperanza es 2*sigma^2.
          El primero es donde entra el Lema de Stein, que afirma que
          para ruido gaussiano:

              E[ < f(y) , b > ]  =  sigma^2 * E[ div f(y) ]

          Esta es la pieza no trivial. Lo que dice, en palabras, es que
          la correlación entre la salida del filtro y el ruido que no
          conocemos se puede medir a través de la sensibilidad del filtro
          a su propia entrada, que sí podemos medir.

    Juntando todo:

        E[MSE] = E[(1/N)*||f(y)-y||^2] + (2*sigma^2/N)*E[div] - 2*sigma^2
                 + sigma^2

               = E[(1/N)*||f(y)-y||^2] - sigma^2 + (2*sigma^2/N)*E[div]

    Que es la fórmula de SURE. La imagen limpia x desapareció.

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

    En el escenario real no tendrías x, y por eso necesitas SURE. Pero en
    este experimento sí la tenés: partís de la imagen limpia, le agregás
    el ruido vos mismo, y por lo tanto podés comparar. Si SURE está bien
    implementado, la curva de SURE en función del parámetro y la curva del
    MSE real deben superponerse casi exactamente, y sobre todo deben tener
    el mínimo en el mismo lugar.

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