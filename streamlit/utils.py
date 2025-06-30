import base64

def image_to_data_url(uploaded_file):
    """Convert uploaded image file to base64 data URL for Gemini vision input"""
    image_bytes = uploaded_file.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = uploaded_file.type  # e.g., 'image/jpeg'
    return f"data:{mime};base64,{encoded}"
