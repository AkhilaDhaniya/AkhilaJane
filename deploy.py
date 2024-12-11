import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import mediapipe as mp
import opencv_stream as ovs
import cv2 as cv


# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path="model/keypoint_classifier/keypoint_classifier.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# Function to make predictions using the TFLite model
def predict_tflite(interpreter, input_details, output_details, data):
    data = np.expand_dims(data, axis=0).astype(np.float32)  # Adjust as per your model input
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), output  # Assuming a classification model


def preprocess_image(image):
    # Initialize Mediapipe Hand Detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    # Convert image to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process image to detect hand landmarks
    results = hands.process(image_rgb)

    # Extract landmarks if hand is detected
    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])  # x, y, z coordinates

        hands.close()
        return np.array(landmarks).flatten()  # Flatten into 1D array

    hands.close()
    return np.zeros(21 * 3)  # Return zeros if no hand is detected


# Streamlit UI
st.set_page_config(page_title="Hand Sign Recognition", layout="wide")

# Sidebar for About Section
with st.sidebar:
    st.title("About")
    st.write("""
    **Hand Sign Recognition App**  
    This app uses a TensorFlow Lite model to recognize American Sign Language (ASL) hand signs.  
    Upload an image or use your camera to predict the corresponding letter.
    """)

# Title
st.title("Hand Sign Recognition (ASL Translator)")
st.write("Recognize hand signs using a trained machine learning model.")

# Dropdown for input method
input_method = st.selectbox("Choose Input Method", ["Upload Image", "Use Camera"])

# Load the TFLite model
model_path = "model/keypoint_classifier/keypoint_classifier.tflite"  # Adjust if your model file has a different name
interpreter, input_details, output_details = load_tflite_model(model_path)

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format for processing
        image_cv = np.array(image)
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGB2BGR)

        # Preprocess and predict
        features = preprocess_image(image_cv)
        label, probabilities = predict_tflite(interpreter, input_details, output_details, features)
        st.success(f"Predicted Letter: **{chr(65 + label)}**")  # Assuming labels 0-25 for A-Z

elif input_method == "Use Camera":
    st.write("Click 'Start' to begin using your webcam for live predictions.")

    # Using opencv-stream for webcam feed
    camera_button = st.button("Start Webcam")
    if camera_button:
        st.write("Webcam feed is now live!")
        stframe = st.empty()

        # opencv-stream capturing
        capture = ovs.camera_feed(width=640, height=480, source=0)

        # Start webcam stream
        for frame in capture:
            # Preprocess and predict
            features = preprocess_image(frame)
            label, probabilities = predict_tflite(interpreter, input_details, output_details, features)

            # Display results on the frame
            cv.putText(frame, f"Predicted: {chr(65 + label)}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            stframe.image(frame, channels="BGR", use_column_width=True)

