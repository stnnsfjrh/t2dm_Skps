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

EFFICIENTNET_PATH = Path(__file__).parent / "efficientnetb0_best.keras"
MOBILENET_PATH = Path(__file__).parent / "mobilenetv2_best.keras"

CLASS_NAMES = [
    "Non-Diabetes",
    "Diabetes"
]


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_all_models():
    eff_model = keras.models.load_model(
        EFFICIENTNET_PATH,
        compile=False
    )
    mob_model = keras.models.load_model(
        MOBILENET_PATH,
        compile=False
    )
    return eff_model, mob_model


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
# GRAD-CAM GENERIK
# (Bekerja untuk Sub-model maupun Sequential/Functional)
# =========================
def gradcam(model, image):

    batch = image[None, ...]

    # Cek apakah model memiliki sub-model (base model) atau merupakan single graph
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    # 1. Kasus jika arsitektur menggunakan Base Model bertingkat
    if base_model is not None:
        target_layer = last_feature_layer(base_model)
        feature_model = keras.Model(
            base_model.input,
            [target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            # Lewati layer sebelum base_model jika ada
            x = batch
            for layer in model.layers:
                if layer == base_model:
                    break
                x = layer(x, training=False)

            conv_output, features = feature_model(x, training=False)

            # Lewati layer setelah base_model
            x = features
            found_base = False
            for layer in model.layers:
                if found_base:
                    x = layer(x, training=False)
                if layer == base_model:
                    found_base = True

            probability = x[:, 0]
            predicted_label = tf.cast(probability >= THRESHOLD, tf.int32)
            score = tf.where(predicted_label == 1, probability, 1 - probability)

        gradients = tape.gradient(score, conv_output)

    # 2. Kasus jika arsitektur satu grafik datar (Flat Model)
    else:
        target_layer = last_feature_layer(model)
        grad_model = keras.Model(
            model.input,
            [target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(batch, training=False)
            probability = predictions[:, 0]
            predicted_label = tf.cast(probability >= THRESHOLD, tf.int32)
            score = tf.where(predicted_label == 1, probability, 1 - probability)

        gradients = tape.gradient(score, conv_output)

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * weights[:, None, None, :], axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap / (
        tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
        + tf.keras.backend.epsilon()
    )

    return heatmap[0].numpy(), target_layer.name


# =========================
# MEMBUAT HEATMAP DAN OVERLAY
# =========================
def make_overlay(
    image,
    heatmap,
    alpha=0.4
):

    heatmap = tf.image.resize(
        heatmap[..., None],
        IMG_SIZE
    ).numpy().squeeze()

    colored_heatmap = plt.get_cmap("jet")(
        np.clip(heatmap, 0, 1)
    )[..., :3]

    original = np.clip(image / 255.0, 0, 1)

    overlay = np.clip(
        (1 - alpha) * original + alpha * colored_heatmap,
        0,
        1
    )

    return colored_heatmap, overlay


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi T2DM - Komparasi Model",
    page_icon="🔬",
    layout="wide"
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

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.055);
        border: 1px dashed rgba(255, 255, 255, 0.25);
        border-radius: 14px;
    }

    .stButton > button {
        width: 100%;
        height: 48px;
        border: none;
        border-radius: 11px;
        background: linear-gradient(90deg, #1976d2, #e53935);
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
# JUDUL & HEADER
# =========================
st.title("Deteksi T2DM dari Citra Lidah")
st.write(
    "Unggah citra lidah untuk melakukan inferensi perbandingan "
    "menggunakan model **EfficientNet-B0** dan **MobileNetV2**."
)

st.warning(
    "Hasil prediksi model bukan diagnosis medis "
    "dan tidak menggantikan pemeriksaan tenaga kesehatan."
)


# =========================
# UPLOAD GAMBAR
# =========================
uploaded = st.file_uploader(
    "Unggah Citra Lidah",
    type=["jpg", "jpeg", "png"]
)


# =========================
# JIKA GAMBAR DIUPLOAD
# =========================
if uploaded is not None:

    file_bytes = uploaded.getvalue()
    display_img = Image.open(uploaded).convert("RGB")

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.image(
            display_img,
            caption="Citra Masukan",
            use_container_width=True
        )
    with right_col:
        st.write("### Siap untuk Evaluasi")
        st.write(
            "Citra akan diproses secara paralel ke dalam model "
            "EfficientNet-B0 dan MobileNetV2 beserta kalkulasi Grad-CAM masing-masing."
        )
        detect_button = st.button(
            "Jalankan Deteksi Komparasi",
            type="primary",
            use_container_width=True
        )

    # =========================
    # PROSES DETEKSI
    # =========================
    if detect_button:

        with st.spinner("Memproses inferensi pada EfficientNet-B0 dan MobileNetV2..."):

            # 1. Load kedua model
            eff_model, mob_model = load_all_models()

            # 2. Preprocessing gambar
            image = preprocess(file_bytes)

            # 3. Prediksi EfficientNet-B0
            eff_label, eff_conf, _ = predict(eff_model, image)
            eff_heatmap, eff_layer = gradcam(eff_model, image)
            eff_colored, eff_overlay = make_overlay(image.numpy(), eff_heatmap)

            # 4. Prediksi MobileNetV2
            mob_label, mob_conf, _ = predict(mob_model, image)
            mob_heatmap, mob_layer = gradcam(mob_model, image)
            mob_colored, mob_overlay = make_overlay(image.numpy(), mob_heatmap)

        st.markdown("---")

        # ==========================================
        # KOMPARASI 2 KOLOM (EFFICIENTNET vs MOBILENET)
        # ==========================================
        col_eff, col_mob = st.columns(2)

        # -----------------------------
        # KOLOM KIRI: EFFICIENTNET-B0
        # -----------------------------
        with col_eff:
            st.subheader("EfficientNet-B0")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Prediksi", CLASS_NAMES[eff_label])
            with m2:
                st.metric("Confidence", f"{eff_conf * 100:.2f}%")

            st.progress(min(max(eff_conf, 0.0), 1.0))

            if eff_label == 1:
                st.error(f"Status: **{CLASS_NAMES[eff_label]}**")
            else:
                st.success(f"Status: **{CLASS_NAMES[eff_label]}**")

            st.write("**Visualisasi Grad-CAM:**")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(
                    eff_colored,
                    caption="Heatmap",
                    use_container_width=True
                )
            with g_col2:
                st.image(
                    eff_overlay,
                    caption="Overlay",
                    use_container_width=True
                )

            st.caption(f"Layer Target: `{eff_layer}`")

        # -----------------------------
        # KOLOM KANAN: MOBILENETV2
        # -----------------------------
        with col_mob:
            st.subheader("MobileNetV2")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Prediksi", CLASS_NAMES[mob_label])
            with m4:
                st.metric("Confidence", f"{mob_conf * 100:.2f}%")

            st.progress(min(max(mob_conf, 0.0), 1.0))

            if mob_label == 1:
                st.error(f"Status: **{CLASS_NAMES[mob_label]}**")
            else:
                st.success(f"Status: **{CLASS_NAMES[mob_label]}**")

            st.write("**Visualisasi Grad-CAM:**")
            g_col3, g_col4 = st.columns(2)
            with g_col3:
                st.image(
                    mob_colored,
                    caption="Heatmap",
                    use_container_width=True
                )
            with g_col4:
                st.image(
                    mob_overlay,
                    caption="Overlay",
                    use_container_width=True
                )

            st.caption(f"Layer Target: `{mob_layer}`")

        st.markdown("---")

        # ==========================================
        # RINGKASAN PERBANDINGAN
        # ==========================================
        st.subheader("Ringkasan Komparasi Model")
        
        is_consensus = (eff_label == mob_label)
        if is_consensus:
            st.info(
                f"**Konsensus Model:** Kedua model sepakat memprediksi citra ini sebagai **{CLASS_NAMES[eff_label]}**."
            )
        else:
            st.warning(
                f"**Perbedaan Prediksi:** EfficientNet-B0 memprediksi **{CLASS_NAMES[eff_label]}** ({eff_conf*100:.1f}%), "
                f"sedangkan MobileNetV2 memprediksi **{CLASS_NAMES[mob_label]}** ({mob_conf*100:.1f}%)."
            )
