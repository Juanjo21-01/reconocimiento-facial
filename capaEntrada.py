# Librerías
import cv2 
import os
# import imutils

# Carpeta de entrenamiento
modelo = "NombrePersona"
# modelo = "FotosOTRO" --> para entrenar a otra persona
ruta1 = "C:/Projectos/ReconocimientoFacial/FotosEquipo"
rutaCompleta = ruta1 + "/" + modelo

# Crear carpeta
if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)   

 
# Abrir camara
camara = cv2.VideoCapture(0)

# abrir videos
# camara = cv2.VideoCapture('video.mp4')

#ruidos
ruidos = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Identificador de rostros
id = 0

# Ciclo para capturar la camara
while True:
    respuesta, captura = camara.read()

    # Si no hay respuesta, salir del ciclo
    if respuesta == False: break

    # Bajar resolución de la imagen
    # captura = imutils.resize(captura, width=640)

    # Convertir a escala de grises
    escala_grises = cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)

    # Guardar imagen
    idCaptura = captura.copy()

    # Detectar caras
    cara = ruidos.detectMultiScale(escala_grises, 1.3, 5)

    # Dibujar rectángulos
    for (x, y, e1, e2) in cara:
        cv2.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)

        # Guardar rostro
        rostroCapturado = idCaptura[y:y+e2, x:x+e1]
        rostroCapturado = cv2.resize(rostroCapturado, (160, 160), interpolation=cv2.INTER_CUBIC)

        # Guardar imagen y asignar id
        cv2.imwrite(rutaCompleta+'/imagen_{}.jpg'.format(id), rostroCapturado)
        id = id + 1
        print("Imagen guardada: ", id)

    # Mostrar captura
    cv2.imshow('Resultado del rostro', captura)

    # Cerrar ventana
    if id == 1000:
        break

# Eliminar las ventanas
camara.release()
cv2.destroyAllWindows()