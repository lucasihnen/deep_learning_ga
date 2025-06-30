import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from utils import image_to_data_url

# Load the trained model
@st.cache_resource
def load_model():
    MODEL_PATH = "../model/tomato&potato_disease_classifier_v2_ft.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
# Set class names (should match training)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    #image = np.array(image) / 255.0
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Gemini setup
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDjv5kiOA45O25NPxjp9B60CcOLjBSS5vY'
chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.9)

# Page setup
st.header("CropBot: Your Crop Health Assistant!")
st.subheader("🌿 Is your crop sick?")
st.markdown("Early signs of crop disease appear on leaves.\nBy detecting them in time, we can help farmers prevent huge losses.")
st.markdown("**Upload a picture of your crop here:**")

# Upload widget
uploaded_image = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🩺 Diagnose!"):
        with st.spinner("🧠 Analyzing image with Gemini and ML model..."):
            uploaded_image.seek(0)
            image_data_url = image_to_data_url(uploaded_image)
            # Gemini diagnosis
            gemini_prompt = [
                SystemMessage(content="You are a helpful AI assistant that analyzes images and an expert on crops."),
                HumanMessage(content=[
                    {"type": "text", "text": """
                            You are a botanical expert, able to recognize crops from their leaves. The crop possibilities are three: potato, tomato or pepper bell. After determining which crop it is, you need to be able to classify it within different states available for each crop.
                            For Pepper Bell, the possible labels are: ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
                            For Potato, the label possibilities are: ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                            For Tomato, the labels are: ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
                            If you are not able to classify between these three, you have to return None.

                            **Important Note** The output must be the single word, most likely to be final label of the plant.
                        """
                     },
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ])
            ]
            gemini_response = chat(gemini_prompt)

            # ML prediction
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_label = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.markdown(f"**🧠 Gemini says:** {gemini_response.content}")
        st.markdown(f"**🔮 ML Model Prediction:** {predicted_label} ({confidence:.2f}% confidence)")
