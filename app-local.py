import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="GI Diseases Classifier", layout="centered")
st.title("Gastrointestinal Diseases Classifier")
st.write("Upload an endoscopic image to classify gastrointestinal tract abnormalities (Esophagitis, Polyps, or Ulcerative Colitis).")

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

# Load and cache the trained model to improve performance
@st.cache_resource
def load_classification_model():
    try:
        model = load_model('InceptionResNetV2.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_classification_model()

CLASS_NAMES = ["Esophagitis", "Polyps", "Ulcerative Colitis"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_tensor = tf.convert_to_tensor(img_array)
    return img_tensor

if uploaded_file is not None:
    col1, col2 = st.columns(2)    

    with col1:
        st.subheader("Uploaded Image")
        image_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Classification Results")

        if model:
            processed_img = preprocess_image(img)
            with st.spinner('Making a prediction...'):
                prediction = model.predict(processed_img, verbose=0)
                predicted_class = np.argmax(prediction[0])
                predicted_label = CLASS_NAMES[predicted_class]
                confidence = prediction[0][predicted_class] * 100
                st.success(f"Predicted Class: **{predicted_label}**.")
    
                for class_name, prob in zip(CLASS_NAMES, prediction[0]):
                    percentage = prob * 100
                    st.write(f"{class_name}: {percentage:.2f}%")
                    st.progress(float(prob))
        else:
            st.error("Model could not be loaded. Please check the model file.")