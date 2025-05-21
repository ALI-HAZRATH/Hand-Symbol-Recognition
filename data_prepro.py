

# --- train_model.py ---
import os, pickle, numpy as np, cv2, matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Config
dataset_path = r"C:\Users\HP\Desktop\HAND SIGN\Dataset"
model_save_path = "mobilenetv2_gesture_model.keras"
label_encoder_path = "label_encoder.pkl"
img_size, batch_size = 128, 32
epochs_stage1, epochs_stage2 = 5, 5

# Step 1: Validate images
def validate_dataset(path):
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path): continue
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            try:
                img = cv2.imread(file_path)
                if img is None or img.size == 0: os.remove(file_path)
            except: os.remove(file_path)

validate_dataset(dataset_path)

# Step 2: Data Generators
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True, validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='categorical', subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='categorical', subset='validation', shuffle=False
)

# Step 3: Class Weights
class_weights = dict(enumerate(compute_class_weight(
    class_weight='balanced', classes=np.unique(train_gen.classes), y=train_gen.classes
)))

# Step 4: Model Build
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Step 5: Train
base_model.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=epochs_stage1, class_weight=class_weights)

base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=epochs_stage2, class_weight=class_weights)

# Step 6: Evaluation
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
print(classification_report(val_gen.classes, y_pred, target_names=list(train_gen.class_indices.keys())))

# Step 7: Save
model.save(model_save_path)
with open(label_encoder_path, "wb") as f:
    pickle.dump({v: k for k, v in train_gen.class_indices.items()}, f)
print("✅ Model and label encoder saved.")
