from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# Konfigurasi penelitian
IMG_SIZE = (224, 224)
THRESHOLD = 0.5
MODEL_PATH = Path(__file__).parent / "efficientnetb0_best.keras"
CLASS_NAMES = ["Non-Diabetes", "Diabetes"]


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(file_bytes):
    img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return tf.cast(img, tf.float32)


def predict(model, batch):
    p = float(model.predict(batch, verbose=0).reshape(-1)[0])
    label = int(p >= THRESHOLD)
    confidence = p if label == 1 else 1 - p
    return p, label, confidence


def last_feature_layer(base_model):
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            pass
    raise ValueError("Layer feature map 4D tidak ditemukan.")


def gradcam(model, batch):
    aug = model.get_layer("augmentation_efficientnetb0")
    base = model.get_layer("efficientnetb0_base")
    gap = model.get_layer("gap_efficientnetb0")
    dropout = model.get_layer("dropout_efficientnetb0")
    classifier = model.get_layer("classifier")

    target = last_feature_layer(base)
    feature_model = keras.Model(
        base.input, [target.output, base.output]
    )

    with tf.GradientTape() as tape:
        x = aug(batch, training=False)
        conv, features = feature_model(x, training=False)
        p = classifier(dropout(gap(features), training=False))[:, 0]
        label = tf.cast(p >= THRESHOLD, tf.int32)
        score = tf.where(label == 1, p, 1 - p)

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv * weights[:, None, None, :], axis=-1
    )
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(
        heatmap, axis=(1, 2), keepdims=True
    ) + tf.keras.backend.epsilon()

    return heatmap[0].numpy(), target.name


def make_overlay(img, heatmap, alpha=0.4):
    heatmap = tf.image.resize(
        heatmap[..., None], IMG_SIZE
    ).numpy().squeeze()

    colored = plt.get_cmap("jet")(np.clip(heatmap, 0, 1))[..., :3]
    original = np.clip(img / 255.0, 0, 1)
    overlay = np.clip((1 - alpha) * original + alpha * colored, 0, 1)

    return colored, overlay


# =========================
# STREAMLIT
# =========================
st.set_page_config(
    page_title="Klasifikasi T2DM Citra Lidah",
    page_icon="🔬",
    layout="wide",
)

st.title("Klasifikasi Citra Lidah Diabetes vs Non-Diabetes")
st.write(
    "Aplikasi klasifikasi citra lidah menggunakan CNN EfficientNet-B0 "
    "dengan transfer learning."
)
st.warning(
    "Hasil model bukan diagnosis medis dan tidak menggantikan "
    "pemeriksaan tenaga kesehatan."
)

uploaded = st.file_uploader(
    "Unggah citra lidah", type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = uploaded.getvalue()

    # Tampilkan citra asli
    display_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(display_img, caption="Citra yang diunggah", width=350)

    # Prediksi
    img = preprocess(file_bytes)
    batch = img[None, ...]
    model = load_model()

    p, label, confidence = predict(model, batch)

    st.subheader("Hasil Prediksi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilitas Diabetes", f"{p * 100:.2f}%")
    c2.metric("Prediksi", CLASS_NAMES[label])
    c3.metric("Confidence", f"{confidence * 100:.2f}%")

    # Grad-CAM
    heatmap, layer_name = gradcam(model, batch)
    colored, overlay = make_overlay(img.numpy(), heatmap)

    st.subheader("Grad-CAM")
    st.caption(f"Feature map terakhir: `{layer_name}`")

    g1, g2, g3 = st.columns(3)
    g1.image(np.clip(img.numpy() / 255.0, 0, 1), caption="Original")
    g2.image(colored, caption="Heatmap")
    g3.image(overlay, caption="Overlay")

    st.info(
        "Grad-CAM menunjukkan area yang relatif berkontribusi terhadap "
        "keputusan model, bukan ciri klinis pasti Diabetes."
    )
