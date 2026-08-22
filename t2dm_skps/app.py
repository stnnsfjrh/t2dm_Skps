from pathlib import Path

import matplotlib.pyplot as plt
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

    return label, confidence, probability


# =========================
# MENCARI FEATURE LAYER
# TERAKHIR UNTUK GRAD-CAM
# =========================
def last_feature_layer(base_model):

    for layer in reversed(base_model.layers):

        try:
            if len(layer.output.shape) == 4:
                return layer

        except Exception:
            pass

    raise ValueError(
        "Layer feature map 4D tidak ditemukan."
    )


# =========================
# GRAD-CAM
# =========================
def gradcam(model, image):

    batch = image[None, ...]

    # Mengambil layer dari model
    augmentation = model.get_layer(
        "augmentation_efficientnetb0"
    )

    base_model = model.get_layer(
        "efficientnetb0_base"
    )

    gap = model.get_layer(
        "gap_efficientnetb0"
    )

    dropout = model.get_layer(
        "dropout_efficientnetb0"
    )

    classifier = model.get_layer(
        "classifier"
    )

    # Cari feature map terakhir
    target_layer = last_feature_layer(
        base_model
    )

    # Model sementara untuk mendapatkan:
    # 1. feature map target
    # 2. output EfficientNetB0
    feature_model = keras.Model(
        base_model.input,
        [
            target_layer.output,
            base_model.output
        ]
    )

    # =========================
    # HITUNG GRADIENT
    # =========================
    with tf.GradientTape() as tape:

        x = augmentation(
            batch,
            training=False
        )

        conv_output, features = feature_model(
            x,
            training=False
        )

        x = gap(features)

        x = dropout(
            x,
            training=False
        )

        probability = classifier(x)[:, 0]

        predicted_label = tf.cast(
            probability >= THRESHOLD,
            tf.int32
        )

        # Score disesuaikan dengan kelas
        # yang diprediksi model
        score = tf.where(
            predicted_label == 1,
            probability,
            1 - probability
        )

    # Gradient terhadap feature map
    gradients = tape.gradient(
        score,
        conv_output
    )

    # Bobot setiap channel
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Membentuk heatmap
    heatmap = tf.reduce_sum(
        conv_output *
        weights[:, None, None, :],
        axis=-1
    )

    # Hilangkan nilai negatif
    heatmap = tf.nn.relu(
        heatmap
    )

    # Normalisasi 0-1
    heatmap = heatmap / (
        tf.reduce_max(
            heatmap,
            axis=(1, 2),
            keepdims=True
        )
        +
        tf.keras.backend.epsilon()
    )

    return (
        heatmap[0].numpy(),
        target_layer.name
    )


# =========================
# MEMBUAT HEATMAP DAN OVERLAY
# =========================
def make_overlay(
    image,
    heatmap,
    alpha=0.4
):

    # Resize heatmap sesuai ukuran gambar
    heatmap = tf.image.resize(
        heatmap[..., None],
        IMG_SIZE
    ).numpy().squeeze()

    # Memberikan warna pada heatmap
    colored_heatmap = plt.get_cmap(
        "jet"
    )(
        np.clip(
            heatmap,
            0,
            1
        )
    )[..., :3]

    # Normalisasi gambar asli
    original = np.clip(
        image / 255.0,
        0,
        1
    )

    # Gabungkan gambar asli + heatmap
    overlay = np.clip(
        (1 - alpha) * original
        +
        alpha * colored_heatmap,
        0,
        1
    )

    return colored_heatmap, overlay


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
        max-width: 820px;
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
    # TAMPILKAN GAMBAR INPUT
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
            "Sedang melakukan deteksi dengan EfficientNet-B0..."
        ):

            # Load model
            model = load_model()

            # Preprocessing
            image = preprocess(
                file_bytes
            )

            # Prediksi
            label, confidence, raw_prob = predict(
                model,
                image
            )

            # Grad-CAM
            heatmap, layer_name = gradcam(
                model,
                image
            )

            # Membuat visualisasi
            colored_heatmap, overlay = make_overlay(
                image.numpy(),
                heatmap
            )


        # ==========================================
        # HASIL DETEKSI EFFICIENTNET-B0
        # ==========================================
        st.subheader("Hasil Inferensi EfficientNet-B0")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="Prediksi Kelas",
                value=CLASS_NAMES[label]
            )
        with metric_col2:
            st.metric(
                label="Keyakinan (Confidence)",
                value=f"{confidence * 100:.2f}%"
            )

        st.progress(min(max(confidence, 0.0), 1.0))

        if label == 1:
            st.error(f"Hasil Klasifikasi: **{CLASS_NAMES[label]}**")
        else:
            st.success(f"Hasil Klasifikasi: **{CLASS_NAMES[label]}**")

        st.markdown("---")


        # ==========================================
        # VISUALISASI KOMPARASI CITRA
        # ==========================================
        st.subheader("Analisis Visual & Grad-CAM")
        st.write(
            "Perbandingan citra dari input asli, pemrosesan model, "
            "hingga pemetaan area relevansi menggunakan Grad-CAM."
        )

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.image(
                display_img,
                caption="1. Citra Asli (Upload)",
                use_container_width=True
            )

        with col_b:
            st.image(
                np.clip(image.numpy() / 255.0, 0, 1),
                caption=f"2. Input Model ({IMG_SIZE[0]}x{IMG_SIZE[1]})",
                use_container_width=True
            )

        with col_c:
            st.image(
                colored_heatmap,
                caption="3. Heatmap Grad-CAM",
                use_container_width=True
            )

        with col_d:
            st.image(
                overlay,
                caption="4. Overlay Grad-CAM",
                use_container_width=True
            )

        st.caption(
            f"Feature layer EfficientNet-B0 yang dianalisis: `{layer_name}`"
        )

        st.info(
            "Area berwarna merah atau kuning menunjukkan bagian "
            "yang relatif memberikan kontribusi lebih besar terhadap "
            "keputusan model. Area biru menunjukkan kontribusi yang "
            "relatif lebih rendah. Grad-CAM bukan bukti bahwa area "
            "tersebut merupakan ciri klinis Diabetes."
        )
