from pathlib import Path

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

CLASS_NAMES = [
    "Non-Diabetes",
    "Diabetes"
]


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
# PREPROCESSING GAMBAR
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

    img = tf.cast(img, tf.float32)

    return img


# =========================
# PREDIKSI
# =========================
def predict(model, image):

    batch = image[None, ...]

    prediction = model.predict(
        batch,
        verbose=0
    )

    probability = float(
        prediction.reshape(-1)[0]
    )

    label = int(
        probability >= THRESHOLD
    )

    if label == 1:
        confidence = probability
    else:
        confidence = 1 - probability

    return label, confidence


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi T2DM",
    page_icon="🔬",
    layout="centered"
)


# =========================
# CSS
# =========================
st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(39, 108, 180, 0.45),
                transparent 38%
            ),
            radial-gradient(
                circle at bottom right,
                rgba(170, 33, 61, 0.42),
                transparent 42%
            ),
            #101827;

        color: white;
    }


    .main .block-container {
        max-width: 760px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    .hero {
        text-align: center;
        margin-bottom: 1.6rem;
    }


    .hero h1 {
        margin: 0;

        font-size: 2.25rem;
        font-weight: 750;

        background: linear-gradient(
            90deg,
            #64b5f6,
            #ff6b6b
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .hero p {
        max-width: 590px;

        margin:
            0.8rem auto
            0;

        color: #cbd5e1;

        font-size: 0.96rem;
        line-height: 1.65;
    }


    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    [data-testid="stFileUploaderDropzone"] {
        background:
            rgba(
                255,
                255,
                255,
                0.055
            );

        border:
            1px dashed
            rgba(
                255,
                255,
                255,
                0.25
            );

        border-radius: 14px;
    }


    .stButton > button {
        width: 100%;
        height: 48px;

        border: none;
        border-radius: 11px;

        background: linear-gradient(
            90deg,
            #1976d2,
            #e53935
        );

        color: white;

        font-weight: 650;
    }


    .stButton > button:hover {
        color: white;
        border: none;

        filter: brightness(1.08);
    }


    .result-card {
        margin-top: 1.35rem;

        padding: 1.6rem;

        text-align: center;

        background:
            rgba(
                8,
                20,
                45,
                0.82
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.13
            );

        border-radius: 16px;

        box-shadow:
            0 14px 35px
            rgba(
                0,
                0,
                0,
                0.18
            );
    }


    .result-title {
        color: #94a3b8;

        font-size: 0.78rem;
        font-weight: 700;

        letter-spacing: 0.12em;
    }


    .result-label {
        margin-top: 0.45rem;

        color: white;

        font-size: 2rem;
        font-weight: 750;
    }


    .result-confidence {
        margin-top: 0.25rem;

        font-size: 2.5rem;
        font-weight: 800;

        background: linear-gradient(
            90deg,
            #64b5f6,
            #ff6b6b
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .result-description {
        margin-top: 0.25rem;

        color: #9fb0c7;

        font-size: 0.82rem;
    }


    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# JUDUL
# =========================
st.markdown(

    <div class="hero">

        <h1>
            Deteksi T2DM dari Citra Lidah
        </h1>

        <p>
            Unggah citra lidah, kemudian tekan
            <b>Deteksi</b>
            untuk mendapatkan hasil klasifikasi
            Diabetes atau Non-Diabetes.
        </p>

    </div>
    ,
    unsafe_allow_html=True
)


# =========================
# PERINGATAN
# =========================
st.warning(
    "Hasil prediksi model bukan diagnosis medis "
    "dan tidak menggantikan pemeriksaan tenaga kesehatan."
)


# =========================
# UPLOAD GAMBAR
# =========================
uploaded = st.file_uploader(
    "Unggah Citra Lidah",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================
# JIKA GAMBAR SUDAH DIUPLOAD
# =========================
if uploaded is not None:

    file_bytes = uploaded.getvalue()

    display_img = Image.open(
        uploaded
    ).convert("RGB")


    # Tampilkan gambar
    st.image(
        display_img,
        caption="Citra yang diunggah",
        width=420
    )


    # Tombol deteksi
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

            model = load_model()

            image = preprocess(
                file_bytes
            )

            label, confidence = predict(
                model,
                image
            )


        # =========================
        # HASIL
        # =========================
        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-title">
                    HASIL DETEKSI
                </div>

                <div class="result-label">
                    {CLASS_NAMES[label]}
                </div>

                <div class="result-confidence">
                    {confidence * 100:.2f}%
                </div>

                <div class="result-description">
                    Tingkat keyakinan model
                    terhadap hasil klasifikasi
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )
