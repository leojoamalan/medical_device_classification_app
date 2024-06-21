import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
import os
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the model
model = load_model('device1.h5')

# Define the classes
classes = ['blood pressure set', 'breast pump', 'commode', 'crutch',
           'glucometer', 'oximeter', 'rippled mattress',
           'therapeutic ultrasound machine', 'thermometer']

# Initialize or load the data DataFrame
data = pd.DataFrame(columns=classes)

# Load or create the all_device_values DataFrame
file_path = "all_device_values.csv"
if os.path.exists(file_path):
    all_device_values = pd.read_csv(file_path)
else:
    all_device_values = pd.DataFrame(columns=['Image'] + classes)

def classify_device(image_rgb):
    def preprocess_image(img):
        img = cv2.resize(img, (299, 299))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    processed_img = preprocess_image(image_rgb)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    return classes[predicted_class_index]

def preprocess_and_extract(image_path):
    reader = easyocr.Reader(['en'])
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Unable to read image from path: {image_path}")
        return [], None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])

    sharpened = cv2.filter2D(gray, -1, kernel_sharpening)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glucose_values = []
    device_type = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        if 0.5 < aspect_ratio < 2.0 and area > 1000:
            roi = image_rgb[y:y+h, x:x+w]
            results = reader.readtext(roi)

            for (box, text, prob) in results:
                numeric_text = ''.join(c for c in text if c.isdigit() or c == '.')
                try:
                    value = float(numeric_text)
                    if 20 <= value <= 600:
                        if device_type is None:
                            device_type = classify_device(roi)
                        glucose_values.append(value)
                        break
                except ValueError:
                    continue

    return glucose_values, device_type

def main():
    st.title('Healthcare Device Data Extractor')
    st.write('Upload an image of a healthcare device (e.g., glucometer, oximeter) to extract the relevant values.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())
        glucose_values, device_type = preprocess_and_extract(temp_image_path)
        
        if glucose_values and device_type:
            st.write("### Detected Values:")
            for value in glucose_values:
                st.write(f"Device: {device_type}, Value: {value}")
                new_row = {'Image': uploaded_file.name, **{device_type: value}}
                all_device_values.loc[len(all_device_values)] = new_row

            st.write(all_device_values)
            save_glucose_data()
        else:
            st.error("Unable to detect values. Please try again with a clearer image or a different angle.")

        try:
            os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Failed to delete temporary file {temp_image_path}: {e}")

        if st.button("Download All Device Values as CSV"):
            download_csv()

        if st.button("Clear All Data"):
            clear_data()

def save_glucose_data():
    file_path = "all_device_values.csv"
    all_device_values.to_csv(file_path, index=False)

def download_csv():
    file_path = "all_device_values.csv"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            csv_data = f.read()
        b64 = base64.b64encode(csv_data).decode('utf-8')
        href = f'<a href="data:file/csv;base64,{b64}" download="all_device_values.csv">Download All Device Values CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No values detected yet.")

def clear_data():
    global all_device_values
    all_device_values = pd.DataFrame(columns=['Image'] + classes)
    save_glucose_data()
    st.success("All data cleared successfully.")

if __name__ == "__main__":
    main()
