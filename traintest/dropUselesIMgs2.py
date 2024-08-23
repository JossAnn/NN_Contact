import os

def eliminar_imagenes_no_permitidas(carpeta):
    extensiones_permitidas = {'.jpg', '.jpeg', '.png'}
    
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        
        if os.path.isfile(ruta_archivo):
            extension = os.path.splitext(archivo)[1].lower()
            
            if extension not in extensiones_permitidas:
                os.remove(ruta_archivo)
                print(f'Eliminado: {archivo}')

# Especifica la ruta de la carpeta aquí
ruta_carpeta = 'Switch_Type_ThreeWay'
eliminar_imagenes_no_permitidas(ruta_carpeta)
