import cv2
import os

videos = ['switchelectroM']

for video in videos:
    vidcap = cv2.VideoCapture(video + '.mp4')
    success, image = vidcap.read()
    count = 0
    os.mkdir(video)
    
    while success:
        # Obtener las dimensiones de la imagen
        height, width, _ = image.shape
        
        # Calcular el tamaño del lado del cuadrado (mínimo de ancho y alto)
        min_dim = min(height, width)
        
        # Calcular las coordenadas para recortar la imagen centrada
        top = (height - min_dim) // 2
        bottom = top + min_dim
        left = (width - min_dim) // 2
        right = left + min_dim
        
        # Recortar la imagen al cuadrado
        cropped_image = image[top:bottom, left:right]
        
        # Guardar la imagen recortada como archivo JPEG
        cv2.imwrite(f"{video}/frame{count}.jpg", cropped_image)
        
        # Leer el siguiente frame
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
