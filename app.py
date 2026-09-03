import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

HF_REPO_ID = "PakeezaKhalid/clear-fog-cnn-classifier"
HF_MODEL_FILENAME = "my_model.keras"
IMAGE_SIZE = 256

st.set_page_config(
    page_title="Clear vs Fog Classifier",
    page_icon="🌫️",
    layout="centered"
)

@st.cache_resource
def load_trained_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

model = load_trained_model()

st.title("Clear vs Fog Image Classification")
st.write("Upload an outdoor image to classify the scene as **Clear** or **Fog**.")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed = preprocess_image(image)
    raw = np.asarray(model.predict(processed, verbose=0)[0], dtype=np.float32)

    # The original semester model has 3 outputs although the dataset uses 2 labels.
    # For the demo, use the two intended class outputs and normalize them.
    two_class_scores = raw[:2]

    score_sum = float(np.sum(two_class_scores))
    if score_sum > 0:
        two_class_scores = two_class_scores / score_sum

    class_index = int(np.argmax(two_class_scores))
    class_names = ["Clear", "Fog"]

    prediction = class_names[class_index]
    confidence = float(two_class_scores[class_index])

    st.markdown("---")
    st.subheader("Prediction")

    if prediction == "Clear":
        st.success(f"### {prediction}")
    else:
        st.info(f"### {prediction}")

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.progress(min(max(confidence, 0.0), 1.0))
