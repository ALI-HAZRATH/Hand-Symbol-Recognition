
# ✋ Real-Time Hand Gesture Recognition using MobileNetV2

## 📌 Introduction

This project focuses on detecting and recognizing **hand gestures** in real-time using deep learning. The aim is to provide a fast, lightweight, and accurate system for interpreting static hand signs using a webcam or uploaded image. It can be extended for gesture-based control systems, assistive technologies, or sign-based communication.
---

## 📦 Requirements

You will need:
- Python 3.8+
- A webcam (for real-time recognition)
- A labeled dataset of hand gestures (images in folders by class)

### Python Libraries:
Install them using:
```bash
pip install -r requirements.txt
pip install streamlit mediapipe
```

Libraries used:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- MediaPipe

---

## 🛠️ Usage

### 🔧 1. Prepare the Dataset
Your dataset should look like:
```
Dataset/
├── Gesture1/
│   ├── img1.jpg
│   ├── img2.jpg
├── Gesture2/
│   └── ...
```

### 🧠 2. Train the Model
Use `data_prepro.py` to:
- Validate and clean images
- Apply augmentations
- Train and fine-tune MobileNetV2
- Save model (`.keras`) and label encoder (`.pkl`)

Run:
```bash
python data_prepro.py
```

### 🎥 3. Real-time Prediction
To predict using webcam:
```bash
python main.py
```

Or use the Streamlit UI:
```bash
streamlit run app.py
```

You can choose to upload an image or use webcam directly from the browser.

---

## 🔄 Pipeline Explanation

1. **Dataset Validation**: Removes unreadable or corrupt images.
2. **Augmentation**: Applies image transformations like flip, rotation, zoom for robust training.
3. **Model Training**:
   - Uses MobileNetV2 pretrained architecture.
   - Adds classification layers for custom gesture classes.
   - Trains in 2 phases (top layers → fine-tuning full model).
4. **Evaluation**: Outputs accuracy, confusion matrix, and classification report.
5. **Saving**: Stores model and label mapping.
6. **Prediction**:
   - Reads webcam frames.
   - Resizes, preprocesses, and feeds to model.
   - Displays label and confidence on screen.

---

## 🤖 Why MobileNetV2?

- **Lightweight**: Good for real-time on laptops and edge devices.
- **Fast inference**: No lag during webcam use.
- **Accurate**: Pretrained on ImageNet, great at feature extraction.
- **Customizable**: Easy to fine-tune with new gesture data.

---

## 💡 Suggestions for Improvement

- Add **hand tracking** using MediaPipe for focused cropping.
- Extend to **dynamic gestures** using video or sequences.
- Include **voice output** for accessibility.
- Add a **REST API** for integration with web or mobile apps.
- Train with **larger and diverse datasets** for robustness.
- Create **mobile version** using TensorFlow Lite.

---

## ✅ Conclusion

This gesture recognition project is a compact and powerful example of how AI can interpret human hand signs in real-time. Using MobileNetV2, it balances speed and accuracy, making it ideal for educational, accessibility, or interactive tech applications. With webcam support and a web UI, it's both developer and user friendly.
