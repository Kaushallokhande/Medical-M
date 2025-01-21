# Medicine Classifier Flask API

This is a Flask-based API that predicts the class of a given medicine image using a pre-trained TensorFlow model. The model can classify images into categories like 'aspirin', 'ibuprofen', or 'paracetamol'. It also computes the model's accuracy on a validation dataset and returns that information along with the prediction.

## Features
- Predicts the class of a medicine based on the uploaded image.
- Returns the model's current accuracy along with the prediction.
- Supports image uploads through HTTP POST requests.

## Requirements

- Python 3.6 or higher
- Flask
- Flask-CORS
- TensorFlow 2.x
- Pillow (PIL)

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medicine-classifier-api.git
   cd medicine-classifier-api
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained model file (`medicine_classifier_model.keras`) in the project directory.

4. Make sure you have a validation dataset in the directory `./data/validation/` with subdirectories representing each medicine class (e.g., `aspirin`, `ibuprofen`, `paracetamol`).

5. Run the Flask app:

   ```bash
   python app.py
   ```

   This will start the server on `http://127.0.0.1:5000/`.

## API Endpoint

### `POST /predict`

This endpoint accepts a POST request with an image file to classify.

#### Request
- **Content-Type**: `multipart/form-data`
- **Form-data key**: `file`
- **File type**: Image (JPEG, PNG, etc.)

#### Response
A JSON response containing the predicted class and the current model accuracy.

```json
{
  "predicted_class": "aspirin",
  "model_accuracy": 0.92
}
```

### Error Responses
If no file is provided or if an error occurs while processing the image, the API will return an error message with an appropriate HTTP status code.

```json
{
  "error": "No file provided"
}
```

```json
{
  "error": "Failed to process image: <error_message>"
}
```

## Model Training

If you want to retrain the model or use your own dataset, ensure that the images are organized in subdirectories according to their class (e.g., `aspirin/`, `ibuprofen/`, `paracetamol/`). Use TensorFlow's `ImageDataGenerator` to load and train the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
