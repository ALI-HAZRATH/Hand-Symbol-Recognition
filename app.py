
# --- app.py ---
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from PIL import Image
from datetime import datetime

# -------------------- CONFIGURATION --------------------
MODEL_PATH = "mobilenetv2_gesture_model.keras"  # Use .keras instead of .h5 if available
ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = 128
st.set_page_config(page_title="🤟 Gesture Recognition", layout="centered")
# --------------------------------------------------------

# -------------------- LOAD MODEL + LABELS --------------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()
# -------------------------------------------------------------

st.title("🤟 Real-Time Hand Gesture Recognition")
option = st.radio("Choose input type:", ("Upload Image", "Webcam"))

# -------------------- PREPROCESSING --------------------
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)
# -------------------------------------------------------

# -------------------- IMAGE UPLOAD ---------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])
        pred_label = label_encoder[pred_class]
        confidence = np.max(preds[0]) * 100

        st.success(f"Predicted Gesture: **{pred_label}** ({confidence:.2f}%)")
# -------------------------------------------------------

# -------------------- WEBCAM MODE ----------------------
elif option == "Webcam":
    st.info("Click 'Start Webcam' to begin. Press 'Stop Webcam' to end session.")

    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)
    if col1.button("▶ Start Webcam"):
        st.session_state.run_webcam = True
    if col2.button("⏹ Stop Webcam"):
        st.session_state.run_webcam = False

    frame_placeholder = st.empty()
    log = []

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible.")
        else:
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                img_input = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
                img_input = np.expand_dims(img_input, axis=0)

                preds = model.predict(img_input)
                pred_class = np.argmax(preds[0])
                pred_label = label_encoder[pred_class]
                confidence = np.max(preds[0]) * 100

                # Logging
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log.append({"Time": timestamp, "Prediction": pred_label, "Confidence": confidence})

                # Draw prediction on frame
                cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")

            cap.release()

            if log:
                df = pd.DataFrame(log)
                df.to_csv("streamlit_webcam_log.csv", index=False)
                st.success("Session ended. Log saved to `streamlit_webcam_log.csv`.")
# -------------------------------------------------------

# -------------------- FOOTER ---------------------------
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit, TensorFlow, and OpenCV")
# -------------------------------------------------------
