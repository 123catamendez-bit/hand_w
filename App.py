import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random

# Dataset inventado de interpretaciones libres
creative_drawings = [
    "un gatito curioso 🐱",
    "una nube esponjosa ☁️",
    "un cohete despegando 🚀",
    "una flor chiquita 🌸",
    "un pez nadando 🐟",
    "un fantasma simpático 👻",
    "un corazón torcido ❤️",
    "un sol sonriente ☀️",
    "un árbol pequeñito 🌳",
    "una carita feliz 🙂",
]

# Datos curiosos por dígito
fun_facts = {
    0: "El 0 fue inventado en la India y revolucionó las matemáticas. ¡Imagínate un mundo sin ceros!",
    1: "El número 1 es el único número que no se considera ni primo ni compuesto.",
    2: "El 2 es el único número par que además es primo.",
    3: "El 3 es considerado un número de la suerte en muchas culturas.",
    4: "El 4 es el número de las estaciones del año.",
    5: "El 5 aparece en la naturaleza todo el tiempo: estrellas de mar, flores, dedos.",
    6: "El 6 es considerado el número perfecto más pequeño.",
    7: "El 7 aparece en las maravillas del mundo y en los días de la semana.",
    8: "El 8 se asocia con la prosperidad en la cultura china.",
    9: "El 9 es el último número antes de que empiece una nueva decena."
}

# Preprocesamiento SIN cv2
def preprocess_image(image):
    img = image.convert("L")  # Escala de grises
    img = ImageOps.invert(img)  # Invertir (para que sea blanco sobre negro)
    img = img.resize((28,28))  # Redimensionar
    img = np.array(img).astype("float32") / 255.0
    return img.reshape(1,28,28,1)

# Predicción
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    img = preprocess_image(image)
    pred = model.predict(img)
    result = np.argmax(pred[0])
    confidence = np.max(pred[0])
    return result, confidence

# Streamlit UI
st.set_page_config(page_title='Oráculo de Dibujos ✨', layout='wide')
st.title('🎨 Oráculo de Dibujos: ¿Qué ve la máquina en tu trazo?')

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

if st.button('✨ Revelar'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA').convert("RGB")

        res, conf = predictDigit(input_image)

        # Mostrar resultado como dígito si está seguro
        if conf > 0.70:
            st.header(f'🔢 Parece un **{res}** con {conf*100:.1f}% de confianza')
            st.write("📖 Dato curioso:", fun_facts[res])
        else:
            st.header("🤔 No estoy seguro que sea un número...")
        
        # Siempre muestra interpretación libre
        st.subheader("🎨 Interpretación creativa:")
        st.write(f"Esto podría ser {random.choice(creative_drawings)}")

    else:
        st.warning('Por favor dibuja en el canvas un número o un doodle.')
