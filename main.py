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

#Imagen, batch y modelo
imgH, imgW = 200, 200
batchSize = 32
modelo = 'models/model05.h5'#las graficas son de este ultimo modelo

#Es pa eliminar imgenes que no son de esos 3 formatos
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

#pa que no se vuelva entrenar si existe
if os.path.exists(modelo):
    model = tf.keras.models.load_model(modelo)
    print("Aqui ta.")
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
        layers.Dropout(0.5),  #desactiva 1/2 neuronas (pa que no se sobreajuste)
        layers.Dense(trainGenerator.num_classes, activation='softmax') #Ajustar clases dinamicamente
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#Esto viene en todos los modelos

    #Esto es para que si la validacion no mejora en 3 generaciones, deje de entrenar
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Entrenae solo si no hay modelo guardado
    if __name__ == '__main__':
        history = model.fit(trainGenerator, epochs=20, validation_data=validGenerator, callbacks=[early_stopping])

        # Visualiza el entrenamiento
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

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
        plt.savefig('TrainValidPlots.png')
        plt.show()

        #Matriz de confusion
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
    camera = cv2.VideoCapture(0)  #Este es el numero de la camara, pero como es predeterminada es cero
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocesamiento de la imagen
            img = cv2.resize(frame, (imgW, imgH))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            
            # Prediccion
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)
            class_name = list(trainGenerator.class_indices.keys())[class_idx]
            
            # Mostrar clase predicha en la imagen (nombre de la carpeta del dataset)
            cv2.putText(frame, f'{class_name}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
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
