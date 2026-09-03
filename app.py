import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model from the .keras file
model = tf.keras.models.load_model('./my_model.keras')

# Define the image size to which the uploaded image will be resized
IMAGE_SIZE = 256

# Set up Streamlit interface
st.title('Fog or Smog Image Prediction')

st.write("Upload an image to classify as Fog or Smog:")

# Allow user to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the image
    image = Image.open(uploaded_image)

    # Preprocess the image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize image to model input size
    image = np.array(image)  # Convert to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype('float32') / 255.0  # Normalize image

    # Get model prediction
    predictions = model.predict(image)
    class_index = np.argmax(predictions, axis=1)[0]  # Get index of predicted class
    class_names = ['Clear', 'Fog']  # Replace with actual class names

    # Styling for the predicted class text
    st.markdown(f'<h2 style="text-align: center; color: #2e8b57; font-size: 50px; font-weight: bold;">Prediction: {class_names[class_index]}</h2>', unsafe_allow_html=True)

    # Display results
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
