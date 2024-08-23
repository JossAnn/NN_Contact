import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response

# Imagen, batch y modelo
imgH, imgW = 200, 200
batchSize = 32
modelo = 'model05.h5'

# Función para eliminar imágenes que no son de los formatos válidos
def dropUseless(path):
    imgValidas = ['.jpg', '.jpeg', '.png']
    for root, dirs, files in os.walk(path):
        for file in files:
            if not any(file.lower().endswith(ext) for ext in imgValidas):
                print(f"No sirve: {file}")
                os.remove(os.path.join(root, file))
dropUseless('dataset')

# Configura el generador de datos
trainDTS = ImageDataGenerator(rescale=1./255, validation_split=0.2)

trainGenerator = trainDTS.flow_from_directory('dataset', target_size=(imgH, imgW), batch_size=batchSize, class_mode='categorical', subset='training')
validGenerator = trainDTS.flow_from_directory('dataset', target_size=(imgH, imgW), batch_size=batchSize, class_mode='categorical', subset='validation')

# Carga el modelo si existe, si no, lo entrena
if os.path.exists(modelo):
    model = tf.keras.models.load_model(modelo)
    print("Aquí está.")
else:
    # Definir modelo
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgH, imgW, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Desactiva 1/2 neuronas (para evitar sobreajuste)
        layers.Dense(trainGenerator.num_classes, activation='softmax') # Ajustar clases dinámicamente
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compilación del modelo

    # Entrenamiento del modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    if __name__ == '__main__':
        history = model.fit(trainGenerator, epochs=20, validation_data=validGenerator, callbacks=[early_stopping])

        # Visualización del entrenamiento
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Precisión del entrenamiento')
        plt.plot(epochs_range, val_acc, label='Precisión de validación')
        plt.legend(loc='lower right')
        plt.title('Precisión del Entrenamiento y Validación')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Pérdida del entrenamiento')
        plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
        plt.legend(loc='upper right')
        plt.title('Pérdida del Entrenamiento y Validación')
        plt.savefig('TrainValidPlots.png')
        plt.show()

        # Matriz de confusión
        Y_pred = model.predict(validGenerator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = validGenerator.classes

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validGenerator.class_indices.keys())
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('MatrizConfusion.png')
        plt.show()

        model.save(modelo)
        print("Modelo guardado en el archivo.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', accuracy=model.evaluate(validGenerator)[1])

def gen_frames():
    camera = cv2.VideoCapture(0)  # Este es el número de la cámara, pero como es predeterminada es cero
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocesamiento de la imagen
            img = cv2.resize(frame, (imgW, imgH))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            
            # Predicción
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)
            class_name = list(trainGenerator.class_indices.keys())[class_idx]

            # Extraer dimensiones del frame
            height, width, _ = frame.shape
            
            # Definir cuadro basado en alguna lógica (e.g., centrado)
            start_point = (width // 4, height // 4)  # Cuadro centrado
            end_point = (3 * width // 4, 3 * height // 4)

            # Dibujar el cuadro
            color = (0, 255, 0)  # Verde
            thickness = 2  # Grosor del cuadro
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            # Mostrar clase predicha en la imagen (nombre de la carpeta del dataset)
            cv2.putText(frame, f'{class_name}', (start_point[0] + 10, start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
