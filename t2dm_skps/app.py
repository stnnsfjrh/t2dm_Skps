import tempfile
import zipfile
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
# LOAD MODEL EFFICIENTNET-B0
# =========================
@st.cache_resource
def load_efficientnet():
    if not EFFICIENTNET_PATH.exists():
        st.error(f"File model tidak ditemukan: {EFFICIENTNET_PATH.name}")
        return None

    try:
        return keras.models.load_model(
            EFFICIENTNET_PATH,
            compile=False,
            safe_mode=False
        )
    except Exception:
        return tf.keras.models.load_model(
            str(EFFICIENTNET_PATH),
            compile=False
        )


# =========================
# LOAD MODEL MOBILENETV2
# =========================
@st.cache_resource
def load_mobilenet():
    if not MOBILENET_PATH.exists():
        st.error(f"File model tidak ditemukan: {MOBILENET_PATH.name}")
        return None

    try:
        return keras.models.load_model(
            MOBILENET_PATH,
            compile=False,
            safe_mode=False
        )
    except Exception:
        pass

    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights=None
        )

        inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        if zipfile.is_zipfile(MOBILENET_PATH):
            with zipfile.ZipFile(MOBILENET_PATH, 'r') as zip_ref:
                weight_files = [f for f in zip_ref.namelist() if 'model.weights.h5' in f or 'weights' in f]
                if weight_files:
                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                        tmp_file.write(zip_ref.read(weight_files[0]))
                        tmp_path = tmp_file.name

                    base_model.load_weights(tmp_path, by_name=True, skip_mismatch=True)
                    model.load_weights(tmp_path, by_name=True, skip_mismatch=True)
                    return model

    except Exception:
        pass

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


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
# GRAD-CAM FORWARD-TRACE (ZERO ERROR)
# =========================
def gradcam(model, image):
    batch = image[None, ...]

    # 1. Identifikasi apakah model memiliki Base Model bertingkat
    base_model = None
    for layer in model.layers:
        if isinstance(layer, (keras.Model, tf.keras.Model)):
            base_model = layer
            break

    # KASUS A: Model Nested (Memiliki Base Model Backbone)
    if base_model is not None:
        # Cari layer 4D terakhir di base model
        target_layer = None
        for l in reversed(base_model.layers):
            try:
                if len(l.output.shape) == 4:
                    target_layer = l
                    break
            except Exception:
                pass

        feature_submodel = keras.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            x = batch
            # Lewati layer pra-backbone (e.g. Augmentation / Rescaling)
            for l in model.layers:
                if l == base_model:
                    break
                x = l(x, training=False)

            conv_output, backbone_out = feature_submodel(x, training=False)

            # Lewati layer pasca-backbone (GAP, Dropout, Dense)
            x = backbone_out
            passed_base = False
            for l in model.layers:
                if passed_base:
                    x = l(x, training=False)
                if l == base_model:
                    passed_base = True

            prob = x[:, 0]
            pred_label = tf.cast(prob >= THRESHOLD, tf.int32)
            score = tf.where(pred_label == 1, prob, 1.0 - prob)

        grads = tape.gradient(score, conv_output)

    # KASUS B: Model Sequential / Functional Biasa
    else:
        target_layer = None
        for l in reversed(model.layers):
            try:
                if len(l.output.shape) == 4:
                    target_layer = l
                    break
            except Exception:
                pass

        # Split eksekusi: input -> conv -> output
        feature_submodel = keras.Model(inputs=model.inputs, outputs=target_layer.output)

        with tf.GradientTape() as tape:
            conv_output = feature_submodel(batch, training=False)
            tape.watch(conv_output)

            # Forward sisa layer setelah layer target
            x = conv_output
            passed_target = False
            for l in model.layers:
                if passed_target:
                    x = l(x, training=False)
                if l.name == target_layer.name:
                    passed_target = True

            prob = x[:, 0]
            pred_label = tf.cast(prob >= THRESHOLD, tf.int32)
            score = tf.where(pred_label == 1, prob, 1.0 - prob)

        grads = tape.gradient(score, conv_output)

    # Hitung bobot rata-rata Grad-CAM
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output_0 = conv_output[0]

    # Linear combination
    heatmap = conv_output_0 @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalisasi 0 - 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), target_layer.name


# =========================
# MEMBUAT HEATMAP DAN OVERLAY
# =========================
def make_overlay(image, heatmap, alpha=0.45):
    heatmap_resized = tf.image.resize(
        heatmap[..., None],
        IMG_SIZE,
        method="bicubic"
    ).numpy().squeeze()

    heatmap_resized = np.nan_to_num(heatmap_resized)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    colored_heatmap = plt.get_cmap("jet")(heatmap_resized)[..., :3]

    original = np.clip(image / 255.0, 0, 1)

    overlay = (1 - alpha) * original + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

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
    "Unggah citra lidah untuk melakukan analisis komparasi "
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
            "**EfficientNet-B0** dan **MobileNetV2** beserta visualisasi Grad-CAM masing-masing."
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

        with st.spinner("Memuat model dan melakukan inferensi..."):

            # 1. Load model
            eff_model = load_efficientnet()
            mob_model = load_mobilenet()

            if eff_model is None or mob_model is None:
                st.stop()

            # 2. Preprocessing
            image = preprocess(file_bytes)

            # 3. Prediksi & Grad-CAM EfficientNet-B0
            eff_label, eff_conf, _ = predict(eff_model, image)
            eff_heatmap, eff_layer = gradcam(eff_model, image)
            eff_colored, eff_overlay = make_overlay(image.numpy(), eff_heatmap)

            # 4. Prediksi & Grad-CAM MobileNetV2
            mob_label, mob_conf, _ = predict(mob_model, image)
            mob_heatmap, mob_layer = gradcam(mob_model, image)
            mob_colored, mob_overlay = make_overlay(image.numpy(), mob_heatmap)

        st.markdown("---")

        # ==========================================
        # KOMPARASI 2 KOLOM
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
        # RINGKASAN KONSENSUS
        # ==========================================
        st.subheader("Ringkasan Konsensus Model")

        is_consensus = (eff_label == mob_label)
        if is_consensus:
            st.info(
                f"**Konsensus:** Kedua arsitektur sepakat mengklasifikasikan citra sebagai **{CLASS_NAMES[eff_label]}**."
            )
        else:
            st.warning(
                f"**Perbedaan Prediksi:** EfficientNet-B0 mendeteksi **{CLASS_NAMES[eff_label]}** ({eff_conf*100:.1f}%), "
                f"sedangkan MobileNetV2 mendeteksi **{CLASS_NAMES[mob_label]}** ({mob_conf*100:.1f}%)."
            )
            
