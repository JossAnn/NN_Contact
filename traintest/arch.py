from PIL import Image
import os

def rotate_images_in_folder(folder_path):
    # Extensiones de imágenes soportadas
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # Crear subcarpeta para guardar imágenes rotadas si no existe
    rotated_folder_path = os.path.join(folder_path)
    os.makedirs(rotated_folder_path, exist_ok=True)

    # Recorrer todas las imágenes en la carpeta
    for filename in os.listdir(folder_path):
        # Verificar que el archivo tenga una extensión soportada
        if filename.lower().endswith(supported_extensions):
            try:
                # Abrir la imagen
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path)

                # Rotar y guardar la imagen en las 4 posiciones
                for angle in [90, 180, 270]:
                    rotated_img = img.rotate(angle, expand=True)
                    rotated_img_filename = f"{os.path.splitext(filename)[0]}{angle}.jpg"
                    rotated_img.save(os.path.join(rotated_folder_path, rotated_img_filename))
                
                print(f"Rotated and saved: {filename}")
            except Exception as e:
                print(f"Error rotating {filename}: {e}")

# Carpeta con las imágenes
folder_path = 'type_B'

# Rotar las imágenes
rotate_images_in_folder(folder_path)
