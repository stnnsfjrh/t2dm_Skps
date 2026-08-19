from pathlib import Path
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# =========================
# KONFIGURASI
# =========================
IMG_SIZE = (224, 224)
THRESHOLD = 0.5
MODEL_PATH = Path(__file__).parent / "efficientnetb0_best.keras"

CLASS_NAMES = ["Non-Diabetes", "Diabetes"]


# =========================
# STREAMLIT CONFIG & CUSTOM CSS
# =========================
st.set_page_config(
    page_title="Deteksi T2DM",
    layout="centered"
)

# Background gradasi biru ke merah & perataan teks ke tengah
st.markdown(
  
    <style>
    /* Background aplikasi gradasi biru ke merah */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 35%, #9a1f40 70%, #d31027 100%);
        background-attachment: fixed;
        color: #ffffff;
    }

    /* Memusatkan teks judul dan paragraf */
    h1, h2, h3, p, label, .stMarkdown {
        text-align: center !important;
        color: #ffffff !important;
    }

    /* Memusatkan gambar yang diunggah */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin: 0 auto;
    }

    /* Memusatkan file uploader widget */
    [data-testid="stFileUploader"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Penyesuaian metrik agar rata tengah */
    [data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    </style>
    ,
    unsafe_allow_html=True
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)


# =========================
# PREPROCESSING
# =========================
def preprocess(file_bytes):
    img = tf.io.decode_image(
        file_bytes,
        channels=3,
        expand_animations=False
    )

    img.set_shape([None, None, 3])

    img = tf.image.resize(
        img,
        IMG_SIZE,
        method="bilinear"
    )

    return tf.cast(img, tf.float32)


# =========================
# PREDIKSI
# =========================
def predict(model, batch):
    prediction = model.predict(
        batch,
        verbose=0
    )

    p = float(prediction.reshape(-1)[0])
    label = int(p >= THRESHOLD)
    confidence = p if label == 1 else 1 - p

    return p, label, confidence


# =========================
# HEADER
# =========================
st.title("Deteksi T2DM dari Citra Lidah")

st.write(
    "Unggah citra lidah kemudian tekan tombol "
    "**Deteksi** untuk mendapatkan hasil prediksi."
)

st.warning(
    "Hasil prediksi model bukan diagnosis medis "
    "dan tidak menggantikan pemeriksaan tenaga kesehatan."
)


# =========================
# UPLOAD GAMBAR
# =========================
uploaded = st.file_uploader(
    "Unggah citra lidah",
    type=["jpg", "jpeg", "png"]
)


# =========================
# TAMPILKAN GAMBAR
# =========================
if uploaded is not None:
    file_bytes = uploaded.getvalue()

    display_img = Image.open(
        io.BytesIO(file_bytes)
    ).convert("RGB")

    st.image(
        display_img,
        caption="Citra yang diunggah",
        width=300
    )

    # =========================
    # TOMBOL DETEKSI
    # =========================
    detect_button = st.button(
        "🔍 Deteksi",
        type="primary",
        use_container_width=True
    )

    # =========================
    # PROSES DETEKSI
    # =========================
    if detect_button:
        with st.spinner("Sedang melakukan deteksi..."):
            model = load_model()
            img = preprocess(file_bytes)
            batch = img[None, ...]
            p, label, confidence = predict(model, batch)

        # =========================
        # HASIL
        # =========================
        st.divider()
        st.subheader("Hasil Deteksi")

        if label == 1:
            st.error(
                f"### Diabetes\n"
                f"Confidence: **{confidence * 100:.2f}%**"
            )
        else:
            st.success(
                f"### Non-Diabetes\n"
                f"Confidence: **{confidence * 100:.2f}%**"
            )

        # =========================
        # PERSENTASE
        # =========================
        st.progress(int(confidence * 100))
        st.metric(
            "Persentase Keyakinan Deteksi",
            f"{confidence * 100:.2f}%"
        )
