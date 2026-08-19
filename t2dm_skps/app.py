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

    img = tf.cast(
        img,
        tf.float32
    )

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
# CSS TAMPILAN
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
st.title(
    "Deteksi T2DM dari Citra Lidah"
)

st.write(
    "Unggah citra lidah, kemudian tekan tombol **Deteksi** "
    "untuk mendapatkan hasil klasifikasi Diabetes atau Non-Diabetes."
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


    # =========================
    # TAMPILKAN GAMBAR
    # =========================
    st.image(
        display_img,
        caption="Citra yang diunggah",
        width=420
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

            model = load_model()

            image = preprocess(
                file_bytes
            )

            label, confidence = predict(
                model,
                image
            )


        # =========================
        # HASIL DETEKSI
        # =========================
        st.subheader(
            "Hasil Deteksi"
        )

        st.success(
            f"Hasil Klasifikasi: {CLASS_NAMES[label]}"
        )

        st.write(
            f"**Tingkat Keyakinan Model: "
            f"{confidence * 100:.2f}%**"
        )
