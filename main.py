import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the pre-trained model (make sure the path to the saved model file is correct)
model = load_model('stroke_detection_model.keras')  # replace with the path to your model file

# Set the image size as used during model training
img_size = (224, 224)

# Streamlit page configuration
st.set_page_config(page_title="Stroke Detection", layout="wide")

# HTML and CSS for toast notification
st.markdown("""
    <style>
        #toast {
            visibility: hidden;
            min-width: 250px;
            margin-left: -125px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 2px;
            padding: 16px;
            position: fixed;
            z-index: 1;
            left: 50%;
            bottom: 30px;
            font-size: 17px;
        }

        #toast.show {
            visibility: visible;
            animation: fadein 0.5s, fadeout 0.5s 2.5s;
        }

        @keyframes fadein {
            from {bottom: 0; opacity: 0;} 
            to {bottom: 30px; opacity: 1;}
        }

        @keyframes fadeout {
            from {bottom: 30px; opacity: 1;} 
            to {bottom: 0; opacity: 0;}
        }
    </style>
    <div id="toast">This is a toast message!</div>
    <script>
        function showToast(message) {
            var toast = document.getElementById("toast");
            toast.innerText = message;  // Set the message for the toast
            toast.className = "show";
            setTimeout(function(){ toast.className = toast.className.replace("show", ""); }, 3000);
        }
    </script>
""", unsafe_allow_html=True)

# Centering the entire app using container
with st.container():
    # Header
    st.title("Brain Stroke Detection")
    st.write("Upload an MRI/CT image, and let AI predict if it shows signs of a stroke or is normal.")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Prediction function
    def predict_image(img):
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Resize and preprocess the image
        img = img.resize(img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to match training process

        # Make the prediction
        prediction = model.predict(img_array)
        return "Stroke Detected" if prediction[0] > 0.5 else "Normal"

    # Display uploaded image and prediction result
    if uploaded_file is not None:
        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Resize the displayed image to Instagram post size (e.g., 300x300 pixels)
        img_display_size = (300, 300)  # Adjust as needed for aesthetics
        img_resized = img.resize(img_display_size)

        # Center the image
        st.image(img_resized, caption='Uploaded Image', use_column_width=False)

        # Create a centered button directly below the image
        if st.button("Predict", key="predict_button"):
            result = predict_image(img)
            st.write(f"**Prediction: {result}**", unsafe_allow_html=True)

            # Show toast notification with the result
            toast_message = "Stroke Detected! Please consult a medical professional." if result == "Stroke Detected" else "No signs of stroke detected."
            st.markdown(f"<script>showToast('{toast_message}');</script>", unsafe_allow_html=True)

            # Optionally add a visual cue based on prediction
            if result == "Stroke Detected":
                st.markdown("<h5 style='color: red;'>Please consult a medical professional.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='color: green;'>No signs of stroke detected.</h5>", unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("Made with ❤️ for health awareness.")
