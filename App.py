import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random

# Datos curiosos por número
fun_facts = {
    0: "El 0 fue inventado en la India y revolucionó las matemáticas. ¡Imagínate un mundo sin ceros!",
    1: "El número 1 es el único número que no se considera ni primo ni compuesto.",
    2: "El 2 es el único número par que además es primo.",
    3: "El 3 es considerado un número de la suerte en muchas culturas, incluso sagrado en varias religiones.",
    4: "El 4 es el número de las estaciones del año y también de los elementos clásicos: tierra, agua, aire y fuego.",
    5: "El 5 aparece en la naturaleza todo el tiempo: las estrellas de mar, las flores y hasta en nuestros dedos.",
    6: "El 6 es considerado el número perfecto más pequeño porque sus divisores suman exactamente 6.",
    7: "El 7 es el número favorito de muchas personas y aparece en las maravillas del mundo y en los días de la semana.",
    8: "El 8 se asocia con la prosperidad en la cultura china y tiene simetría perfecta.",
    9: "El 9 es el último número antes de que empiece una nueva decena: el final de un ciclo."
}

# Posibles interpretaciones creativas
creative_meanings = {
    0: ["¿Dibujaste un huevito? 🥚", "Eso parece un anillo misterioso 💍"],
    1: ["Me recuerda a una vela encendida 🕯️", "Eso se parece a un lápiz solitario ✏️"],
    2: ["Parece un cisne elegante 🦢", "O tal vez una serpiente curiosa 🐍"],
    3: ["Me da la vibra de unas gafas rotas 👓", "O parecen dos montañitas ⛰️"],
    4: ["Parece un estandarte medieval 🏰", "O quizás una silla rara 💺"],
    5: ["¿Dibujaste un anzuelo de pescador? 🎣", "También parece una S traviesa 🌀"],
    6: ["Eso es un caracol feliz 🐌", "O podría ser una espiral galáctica 🌌"],
    7: ["Parece un boomerang 🪃", "O una colina con estilo ⛰️"],
    8: ["Eso es un muñeco de nieve derretido ⛄", "O unos binoculares chiquitos 🔭"],
    9: ["Parece un globo escapando 🎈", "O un signo misterioso al revés 🔄"],
}

# Modelo
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Oráculo de los Números ✨', layout='wide')
st.title('🔮 Oráculo de los Números: ¿Qué dibujaste realmente?')
st.subheader("Dibuja un dígito en el panel y descubre qué significa...")

# Canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Botón
if st.button('✨ Revelar'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        
        st.header(f'🔢 El dígito es: **{res}**')
        st.write("📖 Dato curioso:", fun_facts[res])
        st.write("🤔 También podría ser:", random.choice(creative_meanings[res]))
    else:
        st.warning('Por favor dibuja en el canvas un número antes de presionar el botón.')

# Sidebar
st.sidebar.title("Acerca de:")
st.sidebar.text("Esta app no solo reconoce dígitos,")
st.sidebar.text("¡también los interpreta con un toque creativo!")
st.sidebar.text("¿Será que tu número esconde un secreto?")
