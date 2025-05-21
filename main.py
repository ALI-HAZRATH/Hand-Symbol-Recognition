# # --- predict_webcam.py ---
# import cv2
# import pickle
# import numpy as np
# import tensorflow as tf
# from datetime import datetime
# import pandas as pd

# model = tf.keras.models.load_model("mobilenetv2_gesture_model.keras")
# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# img_size = 128
# cap = cv2.VideoCapture(0)
# print("🎥 Starting webcam... Press 'q' to quit.")

# log = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = cv2.resize(frame, (img_size, img_size))
#     img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)
#     pred_class = np.argmax(preds[0])
#     pred_label = label_encoder[pred_class]
#     confidence = np.max(preds[0]) * 100

#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     log.append({"Time": timestamp, "Prediction": pred_label, "Confidence": confidence})

#     cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
#     cv2.imshow("Hand Gesture Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# pd.DataFrame(log).to_csv("webcam_predictions_log.csv", index=False)
# print("✅ Predictions saved to webcam_predictions_log.csv")



# --- realtime_predict.py ---
import cv2, numpy as np, pickle, tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

IMG_SIZE, NUM_CLASSES = 128, 10

# Load model
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights("mobilenetv2_gesture_model.keras")

# Load labels
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_input = preprocess_input(np.expand_dims(img.astype("float32"), axis=0))
    preds = model.predict(img_input)
    class_idx = np.argmax(preds)
    label = label_encoder[class_idx]
    confidence = preds[0][class_idx]

    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


