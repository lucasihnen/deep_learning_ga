import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO
from fpdf import FPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from utils import image_to_data_url, preprocess_image

# Config
CONFIDENCE_THRESHOLD = 70.0
IMAGE_FOLDER = "resources/images"

# Load the trained model
@st.cache_resource
def load_model():
    MODEL_PATH = "../model/tomato&potato_disease_classifier_v2_ft.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class names (must match training order)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']

# Gemini setup
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDjv5kiOA45O25NPxjp9B60CcOLjBSS5vY'  
chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.9)

# PDF generation function
def generate_claim_pdf(crop_type, diagnosis, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Crop Insurance Claim Report")

    pdf.cell(200, 10, txt="Crop Insurance Claim Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=(
        f"Claimant: Farmer Name\n"
        f"Crop Type: {crop_type}\n"
        f"Detected Condition: {diagnosis}\n"
        f"Model Confidence: {confidence:.2f}%\n"
        f"\n"
        f"Description:\n"
        f"An AI-based leaf image diagnostic system has detected signs of crop disease.\n"
        f"This report certifies the detection result for insurance claim submission.\n"
        f"The condition identified is not classified as healthy and may require agronomic intervention."
    ))

    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

# Streamlit UI
st.header("CropBot: Your Crop Health Assistant!")
st.subheader("🌿 Is your crop sick?")
st.markdown("Early signs of crop disease appear on leaves.\nBy detecting them in time, we can help farmers prevent huge losses.")

# Image selection (gallery-style dropdown)
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_file = st.selectbox("Choose a crop image from gallery:", image_files)

if selected_file:
    image_path = os.path.join(IMAGE_FOLDER, selected_file)
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Selected Image: {selected_file}", use_container_width=True)

    if st.button("🩺 Diagnose!"):
        with st.spinner("🔎 Analyzing image with ML model..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_label = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.markdown(f"**🔮 ML Model Prediction:** {predicted_label} ({confidence:.2f}% confidence)")

        if confidence < CONFIDENCE_THRESHOLD:
            with st.spinner("🤔 Model uncertain — asking Gemini for a second opinion..."):
                with open(image_path, "rb") as img_file:
                    image_data_url = image_to_data_url(img_file)

                gemini_prompt = [
                    SystemMessage(content="You are a helpful AI assistant that analyzes images and an expert on crops."),
                    HumanMessage(content=[
                        {"type": "text", "text": """
                            You are a botanical expert, able to recognize crops from their leaves. The crop possibilities are: potato or tomato.
                            Label options:
                            - Potato: ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                            - Tomato: ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
                            Output the most likely label.
                            **Important Note** The output must be one of the labels mentioned before, no text or explanation.
                        """},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ])
                ]
                gemini_response = chat(gemini_prompt)

            st.markdown(f"⚠️ **Model was uncertain (confidence < {CONFIDENCE_THRESHOLD:.0f}%)**")
            st.markdown(f"💬 **Gemini's Second Opinion:** {gemini_response.content}")

        if "healthy" not in predicted_label.lower():
            crop_type = predicted_label.split("_")[0]
            pdf_file = generate_claim_pdf(crop_type, predicted_label, confidence)
            st.download_button("📄 Download Insurance Claim Report", data=pdf_file, file_name="claim_report.pdf")