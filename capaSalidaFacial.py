# Librerías
import cv2
import os

# Carpeta de entrenamiento
dataRuta = 'C:/Projectos/ReconocimientoFacial/FotosEquipo'
listaPersonas = os.listdir(dataRuta)

# Entrenamiento
entrenamientoModelo1 = cv2.face.EigenFaceRecognizer_create()

# Cargar el modelo
entrenamientoModelo1.read('entrenamientoModelo1.xml')

# Ruidos
ruidos = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Abrir camara
camara = cv2.VideoCapture(0)

# Ciclo para capturar la camara
while True:
    respuesta, captura = camara.read()

    # Si no hay respuesta, salir del ciclo
    if respuesta == False: break

    # Convertir a escala de grises
    escala_grises = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)

    # Guardar imagen
    idCaptura = escala_grises.copy()

    # Detectar caras
    cara = ruidos.detectMultiScale(escala_grises, 1.3, 5)

    # Dibujar rectángulos
    for (x, y, e1, e2) in cara:
       # cv2.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)

        # Guardar rostro
        rostroCapturado = idCaptura[y:y+e2, x:x+e1]
        rostroCapturado = cv2.resize(rostroCapturado, (160, 160), interpolation=cv2.INTER_CUBIC)

        # Predecir
        resultado = entrenamientoModelo1.predict(rostroCapturado)

        # 
        cv2.putText(captura, '{}'.format(resultado), (x, y-5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)

        # Nombrar rostro
        if resultado[1] < 18000:
            cv2.putText(captura, '{}'.format(listaPersonas[resultado[0]]), (x, y-20), 2, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            # Mostrar confianza
            cv2.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)

        else:
            cv2.putText(captura, 'DESCONOCIDO', (x, y-20), 2, 1.1, (0,0, 255), 1, cv2.LINE_AA)
            # Mostrar confianza
            cv2.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)
    
    # Mostrar captura de la cámara
    cv2.imshow('Resultado del rostro', captura)

    # Cerrar ventana
    if cv2.waitKey(1) == ord('q'):
        break


# Eliminar las ventanas
camara.release()
cv2.destroyAllWindows()