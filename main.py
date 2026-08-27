from img_utiles import *
from difusion import *
from sure import *
from ruidos import *
from metricas import *

if __name__ == '__main__':
    imagen = leer_imagen_ppm_rgb(ruta_imagen='entrada/im0001.ppm')

    # Usar solo canal verde
    imagen_verde = imagen[:, :, 1:2]
    sigma = 0.1

    semilla_ruido = 42
    semilla_perturbacion = 12345

    imagen_ruidosa = agregar_ruido_gaussiano(imagen=imagen_verde, varianza=sigma**2, seed=semilla_ruido)

    """guardar_imagen_rgb(imagen_rgb=imagen_verde, ruta_salida='salida/im0001_verde.png')

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
    

    """
    SURE Mínimo: 0.00019549961209968378
    Lambda:      0.1
    Iteraciones: 30
    Delta T:     0.05
    """

    
    rango_lambda = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    rango_iteraciones = [10, 20, 30, 40, 50]
    rango_deltaT = [0.01, 0.05, 0.10, 0.15, 0.20]

    resultados = []
    
    for la in rango_lambda:
        for it in rango_iteraciones:
            for dt in rango_deltaT:
                print(f"Combinacion \nlambda:\t\t{la} \niteraciones:\t\t{it} \ndeltaT:\t\t{dt}")

                imagen_procesada = difusion_anisotropica_rgb(imagen=imagen_ruidosa, lambda_ =la, iteraciones=it, delta_t=dt)
    
                caja_negra = lambda img, la=la, it=it, dt=dt: difusion_anisotropica_rgb(imagen=img, lambda_ =la, iteraciones=it, delta_t=dt)
    
                sure = calcular_SURE(imagen_ruidosa=imagen_ruidosa, 
                                     funcion_caja_negra=caja_negra, 
                                     sigma=sigma, 
                                     semilla_perturbacion=semilla_perturbacion)


                psnr = calcular_psnr(imagen_original=imagen_verde, imagen_procesada=imagen_procesada)
                epi = calcular_epi(imagen_original=imagen_verde, imagen_procesada=imagen_procesada)
                ssim = calcular_ssim_rgb(imagen_original=imagen_verde, imagen_procesada=imagen_procesada)
                fsim = calcular_fsim(imagen_original=imagen_verde, imagen_procesada=imagen_procesada)
    
                resultados.append({
                    "sure": sure,
                    "psnr": psnr,
                    "epi": epi,
                    "ssim": ssim,
                    "fsim": fsim,
                    "lambda": la,
                    "iteraciones": it,
                    "deltaT": dt
                })
    
    print(resultados)
    
    # Find the dictionary with the lowest 'sure' value
    mejor_resultado = min(resultados, key=lambda x: x['sure'])
    
    print("--- MEJOR RESULTADO SURE---")
    print(f"SURE Mínimo: {mejor_resultado['sure']}")
    print(f"Lambda:      {mejor_resultado['lambda']}")
    print(f"Iteraciones: {mejor_resultado['iteraciones']}")
    print(f"Delta T:     {mejor_resultado['deltaT']}")


    # --- MEJOR RESULTADO SURE---
    # SURE Mínimo: 0.00019402291446204316
    # Lambda:      0.1
    # Iteraciones: 10
    # Delta T:     0.15
    

    
