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
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


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

    p = float(
        prediction.reshape(-1)[0]
    )

    label = int(
        p >= THRESHOLD
    )

    confidence = (
        p if label == 1
        else 1 - p
    )

    return p, label, confidence


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi T2DM",
    layout="centered"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown(
    
    <style>

    /* =========================
       BACKGROUND
       ========================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #1e3c72 0%,
            #2a5298 35%,
            #9a1f40 70%,
            #d31027 100%
        );

        color: white;
    }


    /* =========================
       MAIN CONTAINER
       ========================= */

    .main .block-container {
        max-width: 760px;
        padding-top: 45px;
        padding-bottom: 50px;
        padding-left: 30px;
        padding-right: 30px;
    }


    /* =========================
       TITLE
       ========================= */

    .custom-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 10px;

        background: linear-gradient(
            90deg,
            #42a5f5,
            #ef5350
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    /* =========================
       DESCRIPTION
       ========================= */

    .custom-description {
        text-align: center;
        color: #e4e8f0;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 30px;
    }


    /* =========================
       WARNING
       ========================= */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* =========================
       UPLOAD TITLE
       ========================= */

    .upload-title {
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 10px;
    }


    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 8px;
    }


    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 10px;
    }


    /* =========================
       BUTTON
       ========================= */

    .stButton > button {
        width: 100%;
        height: 48px;

        margin-top: 10px;

        border-radius: 10px;
        border: none;

        background: linear-gradient(
            90deg,
            #1976d2,
            #e53935
        );

        color: white;

        font-size: 15px;
        font-weight: 600;

        transition: 0.2s;
    }


    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #2196f3,
            #f44336
        );

        color: white;
    }


    /* =========================
       IMAGE
       ========================= */

    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* =========================
       RESULT CARD
       ========================= */

    .result-card {
        background: rgba(
            8,
            20,
            45,
            0.88
        );

        border: 1px solid rgba(
            255,
            255,
            255,
            0.15
        );

        border-radius: 15px;

        padding: 28px;

        margin-top: 20px;

        text-align: center;
    }


    /* =========================
       RESULT TITLE
       ========================= */

    .result-title {
        color: #c5cfdf;
        font-size: 14px;
        margin-bottom: 8px;
    }


    /* =========================
       RESULT LABEL
       ========================= */

    .result-label {
        color: white;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }


    /* =========================
       CONFIDENCE
       ========================= */

    .result-confidence {
        font-size: 42px;
        font-weight: 700;

        background: linear-gradient(
            90deg,
            #42a5f5,
            #ef5350
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    /* =========================
       CONFIDENCE DESCRIPTION
       ========================= */

    .result-description {
        color: #aebbd0;
        font-size: 13px;
        margin-top: 5px;
    }


    /* =========================
       METRIC
       ========================= */

    [data-testid="stMetric"] {
        background: rgba(
            255,
            255,
            255,
            0.07
        );

        border: 1px solid rgba(
            255,
            255,
            255,
            0.12
        );

        border-radius: 12px;

        padding: 15px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;

        color: rgba(
            255,
            255,
            255,
            0.60
        );

        font-size: 12px;

        margin-top: 45px;

        padding-top: 20px;

        border-top: 1px solid rgba(
            255,
            255,
            255,
            0.12
        );
    }


    /* =========================
       STREAMLIT HEADER
       ========================= */

    header {
        background: transparent !important;
    }

    </style>
    ,
    unsafe_allow_html=True
)


# =========================
# HEADER
# =========================

st.markdown(

    <div class="custom-title">
        Deteksi T2DM dari Citra Lidah
    </div>

    <div class="custom-description">
        Unggah citra lidah kemudian tekan tombol
        <b>Deteksi</b> untuk mendapatkan hasil
        klasifikasi Diabetes atau Non-Diabetes.
    </div>
   ,
    unsafe_allow_html=True
)


# =========================
# WARNING
# =========================

st.warning(
    "Hasil prediksi model bukan diagnosis medis "
    "dan tidak menggantikan pemeriksaan tenaga kesehatan."
)


# =========================
# UPLOAD GAMBAR
# =========================

st.markdown(
    """
    <div class="upload-title">
        Unggah Citra Lidah
    </div>
    """,
    unsafe_allow_html=True
)


uploaded = st.file_uploader(
    "Unggah citra lidah",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
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
        width=400
    )


    # =========================
    # TOMBOL DETEKSI
    # =========================

    detect_button = st.button(
        "Deteksi",
        type="primary",
        use_container_width=True
    )


    # =========================
    # PROSES DETEKSI
    # =========================

    if detect_button:

        with st.spinner(
            "Sedang melakukan deteksi..."
        ):

            # Load model
            model = load_model()

            # Preprocessing
            img = preprocess(
                file_bytes
            )

            # Tambahkan batch dimension
            batch = img[None, ...]

            # Prediksi
            p, label, confidence = predict(
                model,
                batch
            )


        # =========================
        # HASIL DETEKSI
        # =========================

        st.markdown(
        
            <div class="result-card">

                <div class="result-title">
                    HASIL DETEKSI
                </div>
        ,
            unsafe_allow_html=True
        )


        # =========================
        # LABEL
        # =========================

        st.markdown(
            
                <div class="result-label">
                    {CLASS_NAMES[label]}
                </div>
          ,
            unsafe_allow_html=True
        )


        # =========================
        # CONFIDENCE
        # =========================

        st.markdown(
                <div class="result-confidence">
                    {confidence * 100:.2f}%
                </div>

                <div class="result-description">
                    Persentase keyakinan model terhadap hasil deteksi
                </div>

            </div>
            ,
            unsafe_allow_html=True
        )


        # =========================
        # PROGRESS
        # =========================

        st.progress(
            int(confidence * 100)
        )


        # =========================
        # METRIC
        # =========================

        st.metric(
            "Persentase Keyakinan Deteksi",
            f"{confidence * 100:.2f}%"
        )


# =========================
# FOOTER
# =========================

st.markdown(
  
    <div class="footer">
        Klasifikasi T2DM Berdasarkan Citra Lidah
    </div>
   ,
    unsafe_allow_html=True
)
