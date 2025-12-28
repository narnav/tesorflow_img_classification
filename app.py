from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from flask import Flask, render_template, request
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/fit", methods=["GET"])
def fit_model_caller():
    fit_model()
    return {"model":"fited"}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        label = request.form.get("label")  # Doggg / Other

        if not file or file.filename == "":
            return "No file selected"

        if label not in ["Doggg", "Other"]:
            return "Invalid label"

        # dataset/Doggg or dataset/Other
        target_dir = os.path.join("dataset", label)
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, file.filename)
        file.save(file_path)

        prediction = class_img(file_path)

        return {
            "file": file.filename,
            "saved_to": target_dir,
            # "prediction": prediction
        }

    return render_template("index.html")




def class_img(img_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    # model = load_model("keras_Model.h5", compile=False)
    model = load_model("keras_Model.h5", compile=False, safe_mode=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(img_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return ({"Class":class_name[2:],"Confidence Score":confidence_score})


# check  class_img - QA
# class_img("./test/plane.jpg")
# class_img("./test/lion.jpg")


def fit_model():
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 5
    DATASET_DIR = "dataset"

    # Load existing model
    model = load_model("keras_Model.h5", compile=False, safe_mode=False)

    # Compile (required before fit)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Image generator
    datagen = ImageDataGenerator(
        rescale=1.0 / 127.5,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    # Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # Save updated model
    model.save("keras_Model.h5")

    print("✅ Model updated successfully")




if __name__ == "__main__":
    app.run(debug=True)