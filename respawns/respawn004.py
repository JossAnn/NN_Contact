#SOLVED: Guarda graficas
#PROBLEMA: Sobreajuste, graficas mal guardadas?



import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response

# Configura los parámetros de la imagen y el batch size
img_height, img_width = 150, 150
batch_size = 32
model_path = 'model04.h5'

# Función para eliminar imágenes no soportadas
def eliminar_imagenes_invalidas(directory):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                print(f"Eliminando archivo no soportado: {file}")
                os.remove(os.path.join(root, file))

# Llama a la función para limpiar el dataset
eliminar_imagenes_invalidas('dataset')  # Reemplaza con la ruta a tu dataset

# Configura el generador de datos
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset',  # Reemplaza con la ruta a tu dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',  # Reemplaza con la ruta a tu dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Verifica si el modelo ya existe
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado desde el archivo.")
else:
    # Define el modelo
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')  # Ajustar número de clases dinámicamente
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrena el modelo solo si no existe un modelo guardado
    if __name__ == '__main__':
        history = model.fit(
            train_generator,
            epochs=20,  # Puedes ajustar el número de épocas
            validation_data=validation_generator
        )

        # Visualiza el entrenamiento
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(20)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Precision del entrenamiento')
        plt.plot(epochs_range, val_acc, label='Precision de validacion')
        plt.legend(loc='lower right')
        plt.title('Precision del Entrenamiento y Validacion')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Perdida del entrenamiento')
        plt.plot(epochs_range, val_loss, label='Perdida de Validacion')
        plt.legend(loc='upper right')
        plt.title('Perdida del Entrenamiento y Validacion')
        plt.show()
        plt.savefig('TrainValidPlots.png')

        # Genera y muestra la matriz de confusión
        Y_pred = model.predict(validation_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = validation_generator.classes

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        plt.savefig('MatizConfusion.png')

        # Guarda el modelo
        model.save(model_path)
        print("Modelo guardado en el archivo.")

# Implementa un servidor web simple con Flask para visualizar las métricas y la cámara
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', accuracy=model.evaluate(validation_generator)[1])

def gen_frames():
    camera = cv2.VideoCapture(0)  # 0 para la cámara web predeterminada
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocesamiento de la imagen
            img = cv2.resize(frame, (img_width, img_height))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            
            # Predicción
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)
            class_name = list(train_generator.class_indices.keys())[class_idx]
            
            # Mostrar clase predicha en la imagen
            cv2.putText(frame, f'Pred: {class_name}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Codificar la imagen para transmitirla
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
