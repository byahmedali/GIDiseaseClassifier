import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import gdown

# Configure the Streamlit page
st.set_page_config(page_title="GI Diseases Classifier", layout="centered")
st.title("Gastrointestinal Diseases Classifier")
st.write("Upload an endoscopic image to classify gastrointestinal tract abnormalities (Esophagitis, Polyps, or Ulcerative Colitis).")

# Information about the model and usage guidelines
with st.expander("About the Model"):
    st.write("""
    This application uses the state-of-the-art InceptionResNetV2 model fine-tuned on endoscopic images to classify three different gastrointestinal tract abnormalities:
    
    1. **Esophagitis**: Inflammation of the esophagus, which can cause difficulty swallowing, chest pain, and heartburn.
    2. **Polyps**: Abnormal tissue growths that can develop into cancer if left untreated.
    3. **Ulcerative Colitis**: A chronic inflammatory bowel disease that causes inflammation and ulcers in the digestive tract.
    
    The model has achieved an impressive 98% accuracy on unseen endoscopic images, making it highly reliable for clinical decision support.
    
    For best results:
    - Upload clear endoscopic images
    - Ensure proper lighting and focus
    - Images should be in JPG, JPEG, or PNG format
    """)

# Google Drive file ID (from your link)
MODEL_FILE_ID = "1sQ_OdYuvOrCfNuVE_bMHhwqwwzQ-GHG7"
MODEL_PATH = "InceptionResNetV2.keras"

@st.cache_resource
def load_classification_model():
    try:
        # Check if model already exists locally
        if not os.path.exists(MODEL_PATH):
            # Construct gdown URL and download
            download_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(download_url, MODEL_PATH, quiet=False)

        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_classification_model()

# Define class names globally
CLASS_NAMES = ["Esophagitis", "Polyps", "Ulcerative Colitis"]

# Image upload interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image preprocessing function to prepare for model input
def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_tensor = tf.convert_to_tensor(img_array)
    return img_tensor

# Process and classify uploaded images
if uploaded_file is not None:
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    # Display the uploaded image
    with col1:
        st.subheader("Uploaded Image")
        image_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        st.image(img, use_container_width=True)

    # Display classification results
    with col2:
        st.subheader("Classification Results")

        if model:
            # Preprocess the image for model input
            processed_img = preprocess_image(img)
            
            # Generate prediction
            with st.spinner('Making a prediction...'):
                prediction = model.predict(processed_img, verbose=0)
                
                # Determine the predicted class
                predicted_class = np.argmax(prediction[0])
                predicted_label = CLASS_NAMES[predicted_class]
                
                # Display the prediction result
                confidence = prediction[0][predicted_class] * 100
                st.success(f"Predicted Class: **{predicted_label}**.")
            
                # Display each class probability as a progress bar
                for class_name, prob in zip(CLASS_NAMES, prediction[0]):
                    # Convert probability to percentage
                    percentage = prob * 100
                    # Display progress bar with class name and percentage
                    st.write(f"{class_name}: {percentage:.2f}%")
                    st.progress(float(prob))
        else:
            st.error("Model could not be loaded. Please check the model file.")