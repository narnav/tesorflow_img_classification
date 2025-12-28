# Image Classification Web App with Teachable Machine Model

This is a simple Flask-based web application for image classification using a Keras model trained with Google's Teachable Machine.

The app allows users to upload an image via a web interface, processes it, and returns the predicted class along with the confidence score using a pre-trained MobileNet-based model exported from Teachable Machine.

## Features

- Upload images through a simple HTML form
- Classify images using a custom-trained Keras model (`keras_Model.h5`)
- Display the predicted class and confidence score
- Saves uploaded images to an `uploads/` folder for debugging

## Requirements

- Python 3.8+
- TensorFlow (2.x recommended)
- Flask
- Pillow
- NumPy

Install dependencies with:

```bash
pip install flask tensorflow pillow numpy

Project Structure

.
├── app.py                  # Main Flask application
├── keras_Model.h5          # Your trained Keras model from Teachable Machine
├── labels.txt              # Class labels (one per line)
├── templates/
│   └── index.html          # HTML upload form
├── uploads/                # Created automatically - stores uploaded images
└── README.md               # This file

Setup and Usage

Place your exported keras_Model.h5 and labels.txt files in the project root directory.
Create a templates folder in the project root and add index.html with the following content:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Classification</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        h1 { text-align: center; }
        form { text-align: center; margin: 40px 0; }
        input[type="file"] { margin: 20px; }
        button { padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Upload an Image for Classification</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <button type="submit">Upload & Classify</button>
    </form>
</body>
</html>

# Run the application: