# 🤖 SignBot ASL Model Trainer & Verifier

This repository contains the training pipeline and verification script for an **American Sign Language (ASL) Recognition** model. The model is trained using the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and exported to TensorFlow Lite (`.tflite`) format for deployment in mobile apps.

---

## 📂 Project Structure

```

signbot\_backend\_models/
├── train\_model.py         # Trains and saves the Keras model & converts to TFLite
├── verify\_tflite.py       # Verifies the exported .tflite model with dummy data
├── asl\_model.keras        # Trained Keras model (ignored in Git)
├── asl\_model.tflite       # TFLite model for mobile deployment (ignored in Git)
├── .gitignore
├── README.md

````

---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install tensorflow opencv-python numpy
````

---

## 🧠 Train the Model

Train and export the ASL model:

```bash
python train_model.py
```

* Model is trained on the `asl_alphabet_train` dataset
* Uses data augmentation and validation split
* Saves:

  * `asl_model.keras` (Keras model)
  * `asl_model.tflite` (quantized TensorFlow Lite model)

> 📁 Update `DATA_DIR` in `train_model.py` to point to your dataset location.

---

## 🧪 Verify TFLite Model

Run the verification script to check if the `.tflite` model loads and performs inference:

```bash
python verify_tflite.py
```

* Loads the `asl_model.tflite` file
* Prints input/output tensor details
* Performs a dummy prediction on random input

---

## 🧾 Dataset

* [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* Contains static images for A–Z hand gestures
* Ideal for training classification models

---

## 📦 .gitignore Highlights

This project ignores large/binary files like:

```gitignore
*.keras
*.tflite
*.log
__pycache__/
*.pyc
```

---

## 🚀 Use in Flutter

The exported `asl_model.tflite` can be used in real-time sign recognition apps via [`tflite_flutter`](https://pub.dev/packages/tflite_flutter) in your Flutter frontend.

---

## 🔗 Repository

GitHub: [https://github.com/obaidullah72/signbot\_backend\_models](https://github.com/obaidullah72/signbot_backend_models)

---

## 🧑‍💻 Author

**Obaidullah Mansoor**
Co-Founder at [ZONEX DEV](https://www.linkedin.com/in/obaidullah72)

---

## 🛡 License

This project is licensed under the MIT License.
