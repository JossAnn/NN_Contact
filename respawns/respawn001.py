#PROBLEMA: Entrena dos veces


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, Response

# Configura los parámetros de la imagen y el batch size
img_height, img_width = 150, 150
batch_size = 32

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

# Define el modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Ajustar número de clases dinámicamente
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrena el modelo
history = model.fit(
    train_generator,
    epochs=10,  # Puedes ajustar el número de épocas
    validation_data=validation_generator
)

# Visualiza el entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)  # Asegúrate de que el número de épocas coincida con el de entrenamiento

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Guarda el modelo
model.save('model.h5')

# Implementa un servidor web simple con Flask para visualizar las métricas y la cámara
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
        <!doctype html>
        <title>Resultados del Entrenamiento</title>
        <h1>Exactitud en el Conjunto de Validación: {{accuracy}}</h1>
        <h2>Cámara en Vivo</h2>
        <img src="/video_feed">
        """, accuracy=val_acc[-1])

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
