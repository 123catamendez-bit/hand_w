import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random

# --- Diccionarios de creatividad ---
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

fun_facts = {
    0: "El 0 fue inventado en la India y revolucionó las matemáticas. ¡Imagínate un mundo sin ceros!",
    1: "El número 1 es el único número que no se considera ni primo ni compuesto.",
    2: "El 2 es el único número par que además es primo.",
    3: "El 3 es considerado un número mágico y creativo en muchas culturas.",
    4: "El 4 simboliza estabilidad: las 4 estaciones, los 4 puntos cardinales.",
    5: "El 5 aparece en la naturaleza: flores, estrellas de mar, y en nuestros dedos.",
    6: "El 6 es un número perfecto porque sus divisores suman 6.",
    7: "El 7 es el número de la suerte, presente en mitos y leyendas.",
    8: "El 8 se asocia con la prosperidad y tiene simetría perfecta.",
    9: "El 9 simboliza cierre de ciclos y la antesala de algo nuevo."
}

oracles = [
    "✨ El oráculo dice: hoy es un buen día para intentar algo nuevo.",
    "🌌 Tu dibujo revela que pronto tendrás una sorpresa inesperada.",
    "🔥 Veo pasión en tu trazo, sigue tu intuición.",
    "💧 Hoy fluye como el agua, no te preocupes por lo que no controlas.",
    "🍀 Tu dibujo trae buena suerte, aprovéchala.",
    "🌙 El oráculo susurra que descanses más, lo necesitas.",
    "🌟 Hay creatividad en ti esperando salir, no la escondas.",
]

# --- Preprocesamiento sin cv2 ---
def preprocess_image(image):
    img = image.convert("L")  # escala de grises
    img = ImageOps.invert(img)  # invertir
    img = img.resize((28,28))  # redimensionar
    img = np.array(img).astype("float32") / 255.0
    return img.reshape(1,28,28,1)

# --- Predicción ---
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    img = preprocess_image(image)
    pred = model.predict(img)
    result = np.argmax(pred[0])
    confidence = np.max(pred[0])
    return result, confidence

# --- Streamlit ---
st.set_page_config(page_title='Oráculo Creativo 🎨🔮', layout='wide')
st.title('🎨 Oráculo Creativo: descubre qué significa tu dibujo')

st.write("👉 Dibuja un número o cualquier cosa, y el oráculo te dirá lo que ve.")

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

        # Si el modelo está seguro del número
        if conf > 0.70:
            st.success(f"🔢 El oráculo ve el número **{res}** (confianza: {conf*100:.1f}%)")
            st.write("📖 Dato curioso:", fun_facts[res])
        else:
            st.warning("🤔 El oráculo no está seguro que sea un número...")

        # Siempre: interpretación creativa
        st.subheader("🎨 Interpretación artística")
        st.write(f"Esto podría ser {random.choice(creative_drawings)}")

        # Y además: mensaje del oráculo
        st.subheader("🔮 Mensaje del Oráculo")
        st.info(random.choice(oracles))

    else:
        st.warning('Por favor dibuja algo en el canvas antes de presionar el botón.')

# Barra lateral personalizada
st.sidebar.title("🎨 Explorador Creativo")
st.sidebar.markdown("Aquí puedes interactuar con tu dibujo y explorar datos curiosos.")

# Opción de modo de dibujo
modo = st.sidebar.radio(
    "Elige un modo de exploración:",
    ["🔢 Reconocer Dígitos", "🎭 Interpretar como Dibujo", "🎲 Dato Curioso"]
)

# Opción para cambiar colores del lienzo
st.sidebar.subheader("🖌️ Personaliza tu lienzo")
stroke_color = st.sidebar.color_picker("Selecciona el color del lápiz", "#FFFFFF")
bg_color = st.sidebar.color_picker("Selecciona el color de fondo", "#000000")

# Slider divertido para el trazo
stroke_width = st.sidebar.slider("✏️ Grosor del lápiz", 1, 30, 15)

# Un dato curioso random
if modo == "🎲 Dato Curioso":
    curiosidades = [
        "El número cero fue inventado en la India hace más de 1500 años.",
        "El 8 acostado es el símbolo del infinito ∞.",
        "En Japón, el 4 se considera un número de mala suerte.",
        "El 7 es el número más popular en el mundo según encuestas.",
        "El 3 aparece en muchísimos símbolos religiosos y culturales."
    ]
    import random
    st.sidebar.info("💡 " + random.choice(curiosidades))

# Créditos pero con estilo
st.sidebar.markdown("👩‍💻 Desarrollado con ❤️ por *Catalina*")



