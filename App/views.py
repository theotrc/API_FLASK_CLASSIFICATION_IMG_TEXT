from io import BytesIO
import os

from flask import request, jsonify
from App import app
from logging import FileHandler, WARNING
from App import model
import numpy as np
import cv2
import base64
from PIL import Image




file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
app.logger.addHandler(file_handler)

@app.route("/")
def home():
    return "api segmentation image pour voiture autonome"

@app.route("/predict", methods=["GET", "POST"])
def predict():

    json = request.get_json()

    if json:
        desc = json.get("description", None)
        image_b64 = json.get("image", None)
        image = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image))
        image_url = "temp_image.png"
        image.save(image_url)
        predicted_category = model.predict_category(image_url, desc)
        os.remove(image_url)  # Delete the temporary image file
        return {"predicted_category": predicted_category}
    else:
        return jsonify({"error": "Invalid input, expected JSON with 'description' and 'image' fields."}), 400

