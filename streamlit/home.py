import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from utils import image_to_data_url

## Setting up the environment
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDjv5kiOA45O25NPxjp9B60CcOLjBSS5vY'
chat = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)

## Function definition (could be moved to a utils.py file)

## Setting up the page
st.header("CropBot: Your Crop Health Assistant!")

st.subheader("🌿 Is your crop sick?")
st.markdown(
    " Early signs of crop disease appear on leaves.\n"
    "By detecting them in time, we can help farmers prevent huge losses."
)

# Instructions:
st.markdown("**Upload a picture of your crop here:**")

# Upload widget
uploaded_image = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Diagnose button
    if st.button("🩺 Diagnose!"):
        # Send to Gemini chat
        with st.spinner("🧠 Asking Gemini what this image contains..."):

            image_data_url = image_to_data_url(uploaded_image)

            # Define prompt template
            prompt = [
                SystemMessage(content="You are a helpful AI assistant that analyzes images and an expert on crops."),
                HumanMessage(content=[
                    {"type": "text", "text": "Analyze what is the most probable crop from the image, based solely on its leaf and state if its healthy or not. I need you to be brief with the explanations. One phrase maximum"},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ])
            ]

            response = chat(prompt)

        # Display response
        st.markdown(f"**🧠 Gemini says:** {response.content}")

