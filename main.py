import csv
import time
from pathlib import Path

from img_utiles import *
from ruidos import *
from difusion import *
from utiles import *
from metricas import *

# Columnas del CSV de resultados. Las celdas que no aplican quedan vacías.
COLUMNAS_CSV = [
    "imagen",
    "tipo_imagen",
    "tipo_ruido",
    "valor_parametro_ruido",
    "gradiente_medio",
    "alpha",
    "lambda0",
    "lambda",
    "delta_t",
    "iteraciones",
    "psnr",
    "epi",
    "ssim",
    "fsim",
]

def formatear_numero(valor, decimales=6):
    """
    Formatea un número para el CSV usando punto como separador decimal.

    Ejemplo: 1.5 significa una parte entera más cinco décimas.
    """
    return f"{valor:.{decimales}f}"

def calcular_metricas_contra_limpia(imagen_limpia, imagen_procesada, caracteristicas_limpia):
    """
    Calcula las 4 métricas (PSNR, EPI, SSIM, FSIM) de una imagen procesada
    comparándola contra la imagen limpia.

    caracteristicas_limpia son las características FSIM (PC y GM) de la
    imagen limpia, calculadas UNA sola vez por imagen y reutilizadas en
    todas las comparaciones para reducir el tiempo de cómputo.

    Retorna un diccionario con los valores ya formateados para el CSV.
    """
    return {
        "psnr": formatear_numero(calcular_psnr(imagen_limpia, imagen_procesada)),
        "epi": formatear_numero(calcular_epi(imagen_limpia, imagen_procesada)),
        "ssim": formatear_numero(calcular_ssim_rgb(imagen_limpia, imagen_procesada)),
        "fsim": formatear_numero(
            calcular_fsim(
                imagen_limpia,
                imagen_procesada,
                caracteristicas_original=caracteristicas_limpia
            )
        ),
    }

def procesamiento_de_imagenes(
    carpeta_entrada,
    carpeta_salida,
    niveles_ruido,
    picos_poisson,
    alphas,
    lambda0=0.1,
    delta_t=0.15,
    iteraciones=35,
    semilla_base=42,
    nombre_csv="resultados.csv",
):
    """
    Ejecuta el pipeline completo sobre todas las imágenes .ppm de la carpeta
    de entrada y escribe los resultados en un único CSV global.

    Pipeline por imagen:
        1. Se lee la imagen limpia, se calcula su gradiente medio (dato de
           referencia) y se escribe su fila. Las métricas quedan vacías
           porque contra sí misma serían triviales.
        2. Por cada combinación (tipo de ruido, valor del parámetro):
            a. Se genera la imagen ruidosa UNA sola vez, con semilla fija,
               así todos los alphas se evalúan sobre la misma realización
               de ruido y la corrida es reproducible.
            b. Se calcula el gradiente medio G de la imagen ruidosa, que es
               el que usa el lambda adaptativo.
            c. Se calculan las métricas contra la limpia y se escribe la
               fila de la imagen ruidosa.
            d. Por cada alpha:
                - lambda = lambda0 * (1 + alpha * G)
                - Se aplica la difusión sobre la imagen ruidosa.
                - Se calculan las métricas contra la limpia y se escribe
                  la fila de la imagen difundida, con todas las columnas.

    El CSV se escribe de forma incremental (flush después de cada fila):
    si la ejecución se corta, los resultados ya calculados quedan
    guardados. Cada corrida crea el archivo desde cero, así que si querés
    conservar una corrida parcial, renombrala antes de volver a ejecutar.

    Formato del CSV: el punto y coma (;) separa columnas y el punto es el
    separador decimal (ej: 1.5). Las celdas sin dato quedan vacías.

    Con los parámetros actuales quedan 217 filas por imagen:
    1 limpia + 18 ruidosas (3 ruidos x 6 valores) + 198 difundidas (18 x 11 alphas).
    """
    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    rutas_imagenes = sorted(carpeta_entrada.glob("*.ppm"))

    if not rutas_imagenes:
        raise FileNotFoundError(
            f"No se encontraron imágenes .ppm en: {carpeta_entrada}"
        )

    # Cada tupla define: nombre del ruido, función que lo genera y lista de
    # valores del parámetro. La función se llama de forma posicional porque
    # el parámetro se llama distinto en cada una (varianza / pico).
    configuraciones_ruido = [
        ("gaussiano", agregar_ruido_gaussiano, niveles_ruido),
        ("speckle", agregar_ruido_speckle, niveles_ruido),
        ("poisson", agregar_ruido_poisson, picos_poisson),
    ]

    # A cada imagen ruidosa le corresponde una semilla distinta, derivada de
    # la base con un contador. Como el orden de los bucles es fijo, cada
    # corrida genera exactamente el mismo ruido (reproducible), pero las
    # realizaciones entre combinaciones son independientes entre sí.
    contador_semilla = 0

    ruta_csv = carpeta_salida / nombre_csv

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo_csv:
        # restval="" hace que toda columna no incluida en la fila quede vacía.
        # delimiter=";" evita conflictos con Excel en configuración regional
        # en español, donde la coma es el separador decimal.
        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=COLUMNAS_CSV,
            restval="",
            delimiter=";"
        )
        escritor.writeheader()
        archivo_csv.flush()

        total_imagenes = len(rutas_imagenes)

        for numero_imagen, ruta_imagen in enumerate(rutas_imagenes, start=1):
            inicio_imagen = time.perf_counter()
            nombre_imagen = ruta_imagen.name

            print(f"[{numero_imagen}/{total_imagenes}] Procesando {nombre_imagen}")

            imagen_limpia = leer_imagen_ppm_rgb(ruta_imagen)

            # Todo el proceso intersousa solo el canal verda (Índice 1 en RGB)
            # Se conserva el eje del canal como (alto, ancho, 1) para reutilizar sin
            # mayores cambios el resto del código
            imagen_limpia = imagen_limpia[:, :, 1:2]

            # G de la limpia: solo como dato de referencia en el CSV.
            gradiente_medio_limpia = calcular_gradiente_medio(imagen_limpia)

            # Características FSIM (PC y GM) de la limpia, calculadas una
            # sola vez por imagen y reutilizadas en todas las comparaciones.
            caracteristicas_limpia = calcular_caracteristicas_fsim(imagen_limpia)

            escritor.writerow({
                "imagen": nombre_imagen,
                "tipo_imagen": "limpia",
                "gradiente_medio": formatear_numero(gradiente_medio_limpia),
            })
            archivo_csv.flush()

            for tipo_ruido, funcion_ruido, valores_parametro in configuraciones_ruido:
                for valor_parametro in valores_parametro:
                    semilla_actual = semilla_base + contador_semilla
                    contador_semilla += 1

                    # La imagen ruidosa se genera UNA sola vez y se reutiliza
                    # para todos los alphas.
                    imagen_ruidosa = funcion_ruido(
                        imagen_limpia,
                        valor_parametro,
                        seed=semilla_actual
                    )

                    # G de la ruidosa: es el que usa el lambda adaptativo.
                    gradiente_medio_ruidosa = calcular_gradiente_medio(imagen_ruidosa)

                    metricas_ruidosa = calcular_metricas_contra_limpia(
                        imagen_limpia,
                        imagen_ruidosa,
                        caracteristicas_limpia
                    )

                    escritor.writerow({
                        "imagen": nombre_imagen,
                        "tipo_imagen": "ruidosa",
                        "tipo_ruido": tipo_ruido,
                        "valor_parametro_ruido": valor_parametro,
                        "gradiente_medio": formatear_numero(gradiente_medio_ruidosa),
                        **metricas_ruidosa,
                    })
                    archivo_csv.flush()

                    for alpha in alphas:
                        lambda_adaptativo = calcular_lambda_adaptativo(
                            lambda0,
                            alpha,
                            gradiente_medio_ruidosa
                        )

                        imagen_difundida = difusion_anisotropica_rgb(
                            imagen_ruidosa,
                            lambda_=lambda_adaptativo,
                            delta_t=delta_t,
                            iteraciones=iteraciones
                        )

                        metricas_difundida = calcular_metricas_contra_limpia(
                            imagen_limpia,
                            imagen_difundida,
                            caracteristicas_limpia
                        )

                        # La fila difundida repite el G de la ruidosa porque
                        # es el que se usó en su lambda: cada fila queda
                        # autocontenida y permite reconstruir
                        # lambda = lambda0 * (1 + alpha * G).
                        escritor.writerow({
                            "imagen": nombre_imagen,
                            "tipo_imagen": "difundida",
                            "tipo_ruido": tipo_ruido,
                            "valor_parametro_ruido": valor_parametro,
                            "gradiente_medio": formatear_numero(gradiente_medio_ruidosa),
                            "alpha": alpha,
                            "lambda0": lambda0,
                            "lambda": formatear_numero(lambda_adaptativo),
                            "delta_t": delta_t,
                            "iteraciones": iteraciones,
                            **metricas_difundida,
                        })
                        archivo_csv.flush()

                    print(f"    {tipo_ruido} ({valor_parametro}): listo, {len(alphas)} alphas.")

            duracion = time.perf_counter() - inicio_imagen
            print(
                f"[{numero_imagen}/{total_imagenes}] "
                f"{nombre_imagen} terminada en {duracion:.1f} s."
            )

    print(f"Procesamiento completo. Resultados en: {ruta_csv}")

if __name__ == '__main__':
    print('Procesando solo canal verde')
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

    # Grilla refinada según el análisis de la primera imagen (im0001):
    # todos los óptimos de PSNR cayeron entre alpha 0 y 30, y de 40 en
    # adelante las métricas quedan planas. Se densifica la zona 0-30 y se
    # deja alpha=50 como ancla para documentar la saturación.
    alphas = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]

    delta_t = 0.15
    iteraciones = 35

    # Semilla base para la generación de ruido: corridas reproducibles.
    semilla_base = 42

    procesamiento_de_imagenes(
        carpeta_entrada=carpeta_entrada,
        carpeta_salida=carpeta_salida,
        niveles_ruido=niveles_ruido,
        picos_poisson=picos_poisson,
        alphas=alphas,
        lambda0=lambda0,
        delta_t=delta_t,
        iteraciones=iteraciones,
        semilla_base=semilla_base,
    )