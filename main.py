from img_utiles import *
from difusion import *
from sure import *
from ruidos import *

if __name__ == '__main__':
    imagen = leer_imagen_ppm_rgb(ruta_imagen='entrada/im0001.ppm')

    # Usar solo canal verde
    imagen_verde = imagen[:, :, 1:2]
    guardar_imagen_rgb(imagen_rgb=imagen_verde, ruta_salida='salida/im0001_verde.png')

    sigma = 0.05

    imagen_ruidosa = agregar_ruido_gaussiano(imagen=imagen_verde, varianza=sigma**2, seed=None)
    guardar_imagen_rgb(imagen_rgb=imagen_ruidosa, ruta_salida='salida/im0001_ruidosa.png')

    imagen_procesada = difusion_anisotropica_rgb(imagen=imagen_ruidosa, lambda_=0.1, delta_t=0.01, iteraciones=30)
    guardar_imagen_rgb(imagen_rgb=imagen_procesada, ruta_salida='salida/im0001_procesada.png')

    caja_negra = lambda img, la=0.1, it=30, dt=0.01: difusion_anisotropica_rgb(imagen=img, lambda_ =la, iteraciones=it, delta_t=dt)

    sure = calcular_SURE(imagen_ruidosa=imagen_ruidosa, funcion_caja_negra=caja_negra, sigma=sigma)
    mse = calcular_mse_real(imagen_procesada=imagen_procesada, imagen_limpia=imagen_verde)

    print(f"SURE:\t{sure} \nMSE:\t{mse}")
    

    """
    SURE Mínimo: 0.00019549961209968378
    Lambda:      0.1
    Iteraciones: 30
    Delta T:     0.05
    """

    """
    rango_lambda = [0.05, 0.075, 0.1, 0.15, 0.2]
        rango_iteraciones = [10, 20, 30, 40, 50]
        rango_deltaT = [0.05, 0.10, 0.15, 0.20, 0.25]
    
        resultados = []
    
        for la in rango_lambda:
            for it in rango_iteraciones:
                for dt in rango_deltaT:
                    print(f"Combinacion \nlambda:\t\t{la} \niteraciones:\t\t{it} \ndeltaT:\t\t{dt}")
    
                    caja_negra = lambda img, la=la, it=it, dt=dt: difusion_anisotropica_rgb(imagen=img, lambda_ =la, iteraciones=it, delta_t=dt)
    
                    sure = calcular_SURE(imagen_ruidosa=imagen_ruidosa, funcion_caja_negra=caja_negra, sigma=sigma)
    
                    resultados.append({
                        "sure": sure,
                        "lambda": la,
                        "iteraciones": it,
                        "deltaT": dt
                    })
    
        print(resultados)
    
        # Find the dictionary with the lowest 'sure' value
        mejor_resultado = min(resultados, key=lambda x: x['sure'])
    
        print("--- MEJOR RESULTADO ---")
        print(f"SURE Mínimo: {mejor_resultado['sure']}")
        print(f"Lambda:      {mejor_resultado['lambda']}")
        print(f"Iteraciones: {mejor_resultado['iteraciones']}")
        print(f"Delta T:     {mejor_resultado['deltaT']}")
    """

    
