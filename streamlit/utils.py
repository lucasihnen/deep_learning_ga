import base64

def image_to_data_url(image_file):
    mime_type = image_file.type  # e.g., 'image/jpeg'
    data = image_file.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
