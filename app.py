from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('medicine_classifier_model.keras')

# Load class names
class_names = ['aspirin', 'ibuprofen', 'paracetamol']  # Replace with actual class names

# Load and preprocess validation dataset to compute accuracy
validation_dir = "./data/validation"

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

def preprocess_image(image):
    # Convert to RGB if the image has an alpha channel or is grayscale
    if image.mode in ['RGBA', 'LA']:
        image = image.convert('RGB')
    elif image.mode == 'L':  # If grayscale, convert to RGB
        image = image.convert('RGB')
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Compute model accuracy
def compute_accuracy():
    loss, accuracy = model.evaluate(validation_dataset)
    return accuracy

# Initialize model accuracy
model_accuracy = compute_accuracy()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and make a prediction
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_names[predicted_class[0]]

        # Return the prediction and accuracy as JSON
        return jsonify({
            "predicted_class": predicted_label,
            "model_accuracy": model_accuracy  # Return model accuracy
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
