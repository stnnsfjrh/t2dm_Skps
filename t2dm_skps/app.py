from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


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


# ==========================================
# PEMBUATAN ARSITEKTUR NATIVE (DARI CELL 7 & 9)
# ==========================================
def build_data_augmentation(name):
    return keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal", seed=42),
            layers.RandomRotation(factor=0.03, fill_mode="reflect", seed=43),
            layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), fill_mode="reflect", seed=44),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=45),
            layers.RandomContrast(factor=0.10, seed=46),
        ],
        name=name
    )


# =========================
# LOAD MODEL EFFICIENTNET-B0
# =========================
@st.cache_resource
def load_efficientnet():
    try:
        return keras.models.load_model(EFFICIENTNET_PATH, compile=False, safe_mode=False)
    except Exception:
        # Rekonstruksi jika deserialisasi gagal
        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
        aug = build_data_augmentation("augmentation_efficientnetb0")
        x = aug(inputs)
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            name="efficientnetb0_base"
        )
        x = base_model(x)
        x = layers.GlobalAveragePooling2D(name="gap_efficientnetb0")(x)
        x = layers.Dropout(0.3, name="dropout_efficientnetb0")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_best")
        model.load_weights(str(EFFICIENTNET_PATH))
        return model


# =========================
# LOAD MODEL MOBILENETV2
# =========================
@st.cache_resource
def load_mobilenet():
    # Rekonstruksi arsitektur persis Cell 9 untuk mengatasi lambda deserialization
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    aug = build_data_augmentation("augmentation_mobilenetv2")
    x = aug(inputs)
    
    preprocess_layer = layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess"
    )
    x = preprocess_layer(x)
    
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        name="mobilenetv2_base"
    )
    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="gap_mobilenetv2")(x)
    x = layers.Dropout(0.3, name="dropout_mobilenetv2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_best")
    
    # Load bobot yang telah dilatih
    model.load_weights(str(MOBILENET_PATH))
    return model


# =========================
# PREPROCESSING GAMBAR
# =========================
def preprocess(file_bytes):
    image = tf.io.decode_image(
        file_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image


# =========================
# CARI FEATURE LAYER 4D TERAKHIR
# (Persis dari Cell 20)
# =========================
def find_last_4d_feature_layer(base_model):
    for layer in reversed(base_model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                return layer
        except Exception:
            continue
    raise ValueError(
        f"Tidak menemukan layer feature map 4D pada model {base_model.name}."
    )


# =========================
# GRAD-CAM DARI FROZEN BASE
# (Persis dari Cell 20)
# =========================
def gradcam_from_frozen_base(
    full_model,
    base_model_name,
    augmentation_layer_name,
    gap_layer_name,
    dropout_layer_name,
    image_batch,
    preprocess_layer_name=None,
    threshold=0.5
):
    base_model = full_model.get_layer(base_model_name)
    target_layer = find_last_4d_feature_layer(base_model)

    feature_model = keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
        name=f"{base_model.name}_gradcam_feature_model"
    )

    augmentation_layer = full_model.get_layer(augmentation_layer_name)
    gap_layer = full_model.get_layer(gap_layer_name)
    dropout_layer = full_model.get_layer(dropout_layer_name)
    classifier_layer = full_model.get_layer("classifier")

    with tf.GradientTape() as tape:
        x = augmentation_layer(image_batch, training=False)

        if preprocess_layer_name is not None:
            preprocess_layer = full_model.get_layer(preprocess_layer_name)
            x = preprocess_layer(x, training=False)

        conv_output, base_output = feature_model(x, training=False)

        x = gap_layer(base_output)
        x = dropout_layer(x, training=False)
        prediction = classifier_layer(x)

        prob_diabetes = prediction[:, 0]
        predicted_class = tf.cast(prob_diabetes >= threshold, tf.int32)

        class_score = tf.where(
            predicted_class == 1,
            prob_diabetes,
            1.0 - prob_diabetes
        )

    gradients = tape.gradient(class_score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)

    max_value = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

    probability = float(prob_diabetes.numpy()[0])
    pred_label = int(predicted_class.numpy()[0])
    confidence = probability if pred_label == 1 else (1.0 - probability)

    return (
        heatmap.numpy()[0],
        target_layer.name,
        probability,
        pred_label,
        confidence
    )


# =========================
# MEMBUAT HEATMAP & OVERLAY
# (Persis dari Cell 20)
# =========================
def create_overlay(original_image, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (original_image.shape[0], original_image.shape[1])
    ).numpy().squeeze()

    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    jet_map = plt.get_cmap("jet")
    colored_heatmap = jet_map(heatmap_resized)[..., :3]

    original_float = original_image.astype(np.float32)
    if original_float.max() > 1.0:
        original_float = original_float / 255.0

    overlay = (1 - alpha) * original_float + alpha * colored_heatmap
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
    /* Mengizinkan konten menyusut dan melebar mengikuti viewport/zoom */
    .stMainBlockContainer, .block-container {
        max-width: 95% !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
    }

    /* Pastikan gambar dan elemen di dalam kolom ikut mengecil proporsional */
    [data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain;
    }

    /* Background & style default */
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
    unsafe_allow_html=True,
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

        with st.spinner("Memuat model dan menjalankan Grad-CAM..."):

            # 1. Load Model
            eff_model = load_efficientnet()
            mob_model = load_mobilenet()

            # 2. Preprocessing tensor gambar
            image_tensor = preprocess(file_bytes)
            image_np = image_tensor.numpy()
            image_batch = tf.expand_dims(image_tensor, axis=0)

            # 3. Grad-CAM EfficientNet-B0
            (
                heatmap_eff,
                layer_eff,
                prob_eff,
                label_eff,
                conf_eff
            ) = gradcam_from_frozen_base(
                full_model=eff_model,
                base_model_name="efficientnetb0_base",
                augmentation_layer_name="augmentation_efficientnetb0",
                preprocess_layer_name=None,
                gap_layer_name="gap_efficientnetb0",
                dropout_layer_name="dropout_efficientnetb0",
                image_batch=image_batch,
                threshold=THRESHOLD
            )
            colored_eff, overlay_eff = create_overlay(image_np, heatmap_eff, alpha=0.4)

            # 4. Grad-CAM MobileNetV2
            (
                heatmap_mob,
                layer_mob,
                prob_mob,
                label_mob,
                conf_mob
            ) = gradcam_from_frozen_base(
                full_model=mob_model,
                base_model_name="mobilenetv2_base",
                augmentation_layer_name="augmentation_mobilenetv2",
                preprocess_layer_name="mobilenetv2_preprocess",
                gap_layer_name="gap_mobilenetv2",
                dropout_layer_name="dropout_mobilenetv2",
                image_batch=image_batch,
                threshold=THRESHOLD
            )
            colored_mob, overlay_mob = create_overlay(image_np, heatmap_mob, alpha=0.4)

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
                st.metric("Prediksi", CLASS_NAMES[label_eff])
            with m2:
                st.metric("Confidence", f"{conf_eff * 100:.2f}%")

            st.progress(min(max(conf_eff, 0.0), 1.0))

            if label_eff == 1:
                st.error(f"Status: **{CLASS_NAMES[label_eff]}** (P={prob_eff:.4f})")
            else:
                st.success(f"Status: **{CLASS_NAMES[label_eff]}** (P={prob_eff:.4f})")

            st.write("**Visualisasi Grad-CAM:**")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(
                    colored_eff,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )
            with g_col2:
                st.image(
                    overlay_eff,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.caption(f"Layer Target: `{layer_eff}`")

        # -----------------------------
        # KOLOM KANAN: MOBILENETV2
        # -----------------------------
        with col_mob:
            st.subheader("MobileNetV2")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Prediksi", CLASS_NAMES[label_mob])
            with m4:
                st.metric("Confidence", f"{conf_mob * 100:.2f}%")

            st.progress(min(max(conf_mob, 0.0), 1.0))

            if label_mob == 1:
                st.error(f"Status: **{CLASS_NAMES[label_mob]}** (P={prob_mob:.4f})")
            else:
                st.success(f"Status: **{CLASS_NAMES[label_mob]}** (P={prob_mob:.4f})")

            st.write("**Visualisasi Grad-CAM:**")
            g_col3, g_col4 = st.columns(2)
            with g_col3:
                st.image(
                    colored_mob,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )
            with g_col4:
                st.image(
                    overlay_mob,
                    caption="Grad-CAM Overlay",
                    use_container_width=True
                )

            st.caption(f"Layer Target: `{layer_mob}`")

        st.markdown("---")

        # ==========================================
        # RINGKASAN KONSENSUS
        # ==========================================
        st.subheader("Ringkasan Konsensus Model")

        if label_eff == label_mob:
            st.info(
                f"**Konsensus:** Kedua arsitektur sepakat mengklasifikasikan citra sebagai **{CLASS_NAMES[label_eff]}**."
            )
        else:
            st.warning(
                f"**Perbedaan Prediksi:** EfficientNet-B0 mendeteksi **{CLASS_NAMES[label_eff]}** ({conf_eff*100:.1f}%), "
                f"sedangkan MobileNetV2 mendeteksi **{CLASS_NAMES[label_mob]}** ({conf_mob*100:.1f}%)."
            )
