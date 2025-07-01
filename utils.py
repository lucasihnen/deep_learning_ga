import base64
import numpy as np
import mimetypes
import base64
from io import BytesIO
from fpdf import FPDF

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