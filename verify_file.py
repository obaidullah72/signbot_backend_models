import numpy as np
import tensorflow as tf
import logging

# -------------------- Configuration --------------------
TFLITE_MODEL_PATH = "asl_model.tflite"
IMG_SIZE = (64, 64)

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_model(tflite_path):
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logging.info("Model loaded successfully.")
        logging.info("Input details: %s", input_details)
        logging.info("Output details: %s", output_details)

        # Create dummy input: shape (1, 64, 64, 1)
        dummy_input = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], dummy_input)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        logging.info("Inference successful. Output shape: %s", output_data.shape)
        logging.info("First prediction: %s", output_data[0])

    except Exception as e:
        logging.error("Error verifying TFLite model: %s", e)

# -------------------- Run Verification --------------------
if __name__ == "__main__":
    verify_model(TFLITE_MODEL_PATH)
