import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

with st.spinner("Cargando Imágenes..."):
    st.markdown('''
        :blue[Modelo de entrenamiento de imágenes con redes convolucionales de cantidad de épocas a elegir] ''')
    st.markdown(" &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

# Nombres de las clases
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Función para cargar las etiquetas desde un archivo CSV
def load_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
    labels_dict = df.set_index('filename')['label'].to_dict()
    return labels_dict

# Función para cargar las imágenes desde una carpeta y asignar etiquetas
def load_images_from_folder(folder, labels_dict=None, img_size=(28, 28)):
    images = []
    labels = []
    nombres = []
    
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convertir a escala de grises
            img = img.resize(img_size)  # Redimensionar la imagen
            img_array = np.array(img) / 255.0  # Normalizar a valores entre 0 y 1
            images.append(img_array)
            
            if labels_dict is not None:
                label = labels_dict[filename.split('.')[0]]  # Obtener la etiqueta desde el diccionario
                
                # Validar las etiquetas
                if int(label) >= 0 and int(label) < 10:
                    labels.append(label)
                else:
                    print('Etiqueta errónea:', label)
            else:
                # Cuando es el de testeo, cargar nombres
                nombres.append(filename.split('.')[0])
                
    if labels_dict is not None:
        return np.array(images), np.array(labels)
    else:
        return np.array(images), nombres

# Funciones para visualizar los resultados
def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):.2f}%")

def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('green')

# Rutas a las carpetas de imágenes y el archivo CSV
train_folder = './test2'
test_folder = './testeo'
labels_csv_path = './train/train.csv'

# Título de la aplicación
st.title(":red[Clasificación de Prendas]")

# Cargar las etiquetas y las imágenes al inicio
with st.spinner("Cargando Imágenes..."):
    labels_dict = load_labels_from_csv(labels_csv_path)
    train_images, train_labels = load_images_from_folder(train_folder, labels_dict)
    test_images, test_nombres = load_images_from_folder(test_folder)
        
    st.success("Carga Completa")

# Añadir canal de color a las imágenes
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Mostrar algunas imágenes de entrenamiento
st.write("### Imágenes de Entrenamiento")
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_images[i].reshape(28, 28), cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(class_names[int(train_labels[i])])
st.pyplot(fig)

# Definir la arquitectura de la CNN
model = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Convertir las etiquetas de cadenas a enteros
train_labels = train_labels.astype(np.int32)
# Personalizar el botón
st.markdown("""
    <style>
    .stButton button {
        background-color: #007BFF;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)
# Controles interactivos para entrenar el modelo
epochs = st.slider("Número de épocas", 1, 20, 10)
if st.button("Entrenar Modelo "):
    with st.spinner("Entrenando..."):
        history = model.fit(train_images, train_labels, epochs=epochs, validation_split=0.2)
        st.title(":red[Graficos de Performance del modelo]")

        # Mostrar gráfica de precisión
        st.write("### Precisión del Modelo")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Precisión en Entrenamiento')
        ax.plot(history.history['val_accuracy'], label='Precisión en Validación')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Precisión')
        ax.legend()
        st.pyplot(fig)

        # Realizar predicciones
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)

    # Mostrar la imagen y el valor predicho en Streamlit
    st.write("### Imágenes y Predicciones")
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, 2*num_cols, figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions)
    st.pyplot(fig)

    # Guardar las predicciones en un archivo CSV
    submission = pd.DataFrame({'id': test_nombres, 'label': predicted_labels})
    submission.to_csv('Predicciones.csv', index=False)
    st.write("Predicciones guardadas en `Predicciones.csv`")
