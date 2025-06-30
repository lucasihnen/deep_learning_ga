import base64
import numpy as np
import mimetypes

def image_to_data_url(file):
    """Convert an image file (from disk or Streamlit upload) to base64 data URL"""
    image_bytes = file.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Determine MIME type safely
    if hasattr(file, "type"):
        mime = file.type  # from Streamlit's uploaded_file
    else:
        # fallback: guess MIME type from file name if available
        if hasattr(file, "name"):
            mime = mimetypes.guess_type(file.name)[0] or "image/jpeg"
        else:
            mime = "image/jpeg"

    return f"data:{mime};base64,{encoded}"

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image
