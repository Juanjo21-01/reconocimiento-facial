# Librerías
import cv2
import os
import numpy as np
import time 

# Carpeta de entrenamiento
dataRuta = 'C:/Projectos/ReconocimientoFacial/FotosEquipo'
listaPersonas = os.listdir(dataRuta)
 
# print('Lista de personas: ', listaPersonas)

# 
ids = []
rostrosData = []
id = 0 
tiempoInicial = time.time()

for nombrePersona in listaPersonas:
    print('Procesando persona: ', nombrePersona)
    rutaPersona = dataRuta + '/' + nombrePersona

    # 
    for nombreRostro in os.listdir(rutaPersona):
        print('Rostro: ', nombreRostro)
        ids.append(id)

        # 
        rostrosData.append(cv2.imread(rutaPersona+'/'+nombreRostro, 0))
        # imagenes = cv2.imread(rutaPersona+'/'+nombreRostro, 0)

    id = id + 1

     # tiempo de espera
    tiempoFinal = time.time()
    tiempoTotalLectura = tiempoFinal - tiempoInicial
    print('Tiempo total de lectura: ', tiempoTotalLectura)

# Entrenamiento
entrenamientoModelo1 = cv2.face.EigenFaceRecognizer_create()

# Entrenar
print('Entrenando...')
entrenamientoModelo1.train(rostrosData, np.array(ids))

tiempoFinalEntrenamiento = time.time()
tiempoTotalEntrenamiento = tiempoFinalEntrenamiento - tiempoTotalLectura
print('Tiempo total de entrenamiento: ', tiempoTotalEntrenamiento)

# Almacenar el modelo
entrenamientoModelo1.write('entrenamientoModelo1.xml')
print('Modelo almacenado...')