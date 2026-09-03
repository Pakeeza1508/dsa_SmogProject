import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Configuration
# --------------------------------------------------

HF_REPO_ID = "PakeezaKhalid/clear-fog-cnn-classifier"
HF_MODEL_FILENAME = "my_model.keras"

IMAGE_SIZE = 256
CLASS_NAMES = ["Clear", "Fog"]

st.set_page_config(
    page_title="Clear vs Fog Classifier",
    page_icon="🌫️",
    layout="centered"
)

# --------------------------------------------------
# Download + load model
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )
    return tf.keras.models.load_model(model_path)

try:
    model = load_trained_model()
except Exception as exc:
    st.error(f"Could not download/load model: {exc}")
    st.stop()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("Clear vs Fog Image Classification")

st.write(
    "Upload an outdoor/environmental image. "
    "The model is loaded from Hugging Face and inference runs through TensorFlow/Keras."
)

uploaded_image = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    x = preprocess_image(image)

    try:
        predictions = np.asarray(model.predict(x, verbose=0)[0], dtype=np.float32)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    output_count = int(predictions.shape[0])

    st.subheader("Model Output")

    # Intended final behavior if model output matches app labels
    if output_count == len(CLASS_NAMES):
        class_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[class_index]
        confidence = float(predictions[class_index])

        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        st.write("### Class Probabilities")
        for name, probability in zip(CLASS_NAMES, predictions):
            st.write(f"- **{name}:** {float(probability) * 100:.2f}%")

    # Safety behavior for current legacy 3-output model
    else:
        st.warning(
            f"The saved model produces {output_count} outputs, "
            f"while the current application has {len(CLASS_NAMES)} intended labels "
            f"({', '.join(CLASS_NAMES)})."
        )

        st.error(
            "A reliable Clear/Fog label cannot be assigned until the model output layer "
            "and class mapping are aligned."
        )

        st.write("### Raw Model Probabilities")
        for index, probability in enumerate(predictions):
            st.write(f"- **Output {index}:** {float(probability) * 100:.2f}%")

        st.caption(
            "This validation prevents the interface from silently mapping an unknown "
            "third output to an incorrect class."
        )

st.info(
    "Limitation: the model is a closed-set environmental classifier. "
    "Unrelated images can still produce model scores and should not be interpreted "
    "as meaningful Clear/Fog predictions."
)
