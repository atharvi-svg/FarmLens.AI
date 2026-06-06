from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
from crop_segment import segment_crop

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

# Create uploads folder only if it doesn't exist
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")

    if file and file.filename:

        # Save original image
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process image
        image = Image.open(filepath).convert("RGB")
        image = np.array(image)

        segmented, healthy, unhealthy = segment_crop(image, k=3)

        # Save segmented image
        seg_filename = "seg_" + file.filename
        segmented_path = os.path.join(UPLOAD_FOLDER, seg_filename)
        Image.fromarray(segmented).save(segmented_path)

        return render_template(
            "index.html",
            original="uploads/" + file.filename,
            segmented="uploads/" + seg_filename,
            healthy=round(healthy, 2),
            unhealthy=round(unhealthy, 2)
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
