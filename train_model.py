import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# -------------------- Configuration --------------------
DATA_DIR = r"F:\SignBot Dataset\asl_alphabet_train\asl_alphabet_train"
IMG_SIZE = (64, 64)
EPOCHS = 20
BATCH_SIZE = 64
MODEL_NAME = "asl_model.keras"
TFLITE_MODEL_NAME = "asl_model.tflite"

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Load Dataset --------------------
def load_dataset(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    label_map = {chr(i): i - 65 for i in range(65, 91)}  # A-Z -> 0-25

    logging.info("Loading dataset from: %s", data_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    for label in sorted(label_map.keys()):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize
                images.append(img.reshape(img_size[0], img_size[1], 1))
                labels.append(label_map[label])
            except Exception as e:
                logging.warning("Failed to load image: %s - %s", img_path, e)

    logging.info("Total images loaded: %d", len(images))
    return np.array(images), np.array(labels)

# -------------------- Build Model --------------------
def build_model(input_shape, num_classes=26):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Main --------------------
def main():
    try:
        X, y = load_dataset(DATA_DIR, IMG_SIZE)
        y = to_categorical(y, num_classes=26)

        # Split manually (if you want better control)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(X_train)

        model = build_model((IMG_SIZE[0], IMG_SIZE[1], 1))

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]

        logging.info("Starting training...")
        model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

        logging.info("Training completed.")

        # Save final model
        model.save(MODEL_NAME)
        logging.info("Saved final model to %s", MODEL_NAME)

        # -------------------- Convert to TFLite --------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optimization (for smaller, faster TFLite model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        with open(TFLITE_MODEL_NAME, "wb") as f:
            f.write(tflite_model)

        logging.info("Converted model to TensorFlow Lite and saved as %s", TFLITE_MODEL_NAME)

    except Exception as e:
        logging.error("An error occurred: %s", e)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
