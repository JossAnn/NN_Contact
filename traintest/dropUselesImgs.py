from PIL import Image
import os

def validate_images(directory):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_extensions:
                print(f"Deleting non-image file: {file_path}")
                os.remove(file_path)
                continue
            try:
                img = Image.open(file_path)
                img.verify()  # Verifica si la imagen está corrupta
            except (IOError, SyntaxError) as e:
                print(f"Deleting corrupt image: {file_path}")
                os.remove(file_path)

# Ruta a la carpeta principal que contiene las subcarpetas con imágenes
dataset_directory = "dataset"
validate_images(dataset_directory)
