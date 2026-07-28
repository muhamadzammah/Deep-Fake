from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2

# Library LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model
MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
model = tf.keras.models.load_model(MODEL_PATH)

# Inisialisasi LIME Explainer
explainer = lime_image.LimeImageExplainer()

# =========================
# CORE FUNCTIONS
# =========================

def preprocess_for_lime(img_path):
    """LIME membutuhkan input image dalam format tertentu"""
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img).astype('double')
    # Normalisasi sesuai EfficientNet (biasanya /255 atau preprocess_input)
    return img_array

def model_predict_function(images):
    """Fungsi wrapper agar LIME bisa membaca output model Flask/Keras"""
    # LIME mengirim batch images, kita harus preprocess tiap image
    processed = tf.keras.applications.efficientnet.preprocess_input(images.copy())
    return model.predict(processed)

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    orig_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(orig_path)

    try:
        # 1. Prediksi Standar
        img_for_pred = preprocess_for_lime(orig_path)
        img_batch = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(img_for_pred, axis=0))
        preds = model.predict(img_batch)
        score = preds[0][0]
        
        label = "REAL" if score > 0.5 else "FAKE"
        confidence = score if score > 0.5 else 1 - score

        # 2. Penjelasan LIME
        # hide_color=0 (hitam) akan menutupi area yang tidak relevan
        explanation = explainer.explain_instance(
            img_for_pred.astype('double'), 
            model_predict_function, 
            top_labels=1, 
            hide_color=0, 
            num_samples=500 # Semakin tinggi semakin akurat tapi lambat
        )

        # Ambil area positif yang mendukung prediksi (label 0 karena biner)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=True
        )

        # Gabungkan batas area (boundaries) ke gambar
        img_bound = mark_boundaries(temp / 255.0, mask)
        img_bound = (img_bound * 255).astype(np.uint8)
        img_bound = cv2.cvtColor(img_bound, cv2.COLOR_RGB2BGR)

        # Simpan hasil LIME
        result_filename = "lime_" + file.filename
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, img_bound)

        return jsonify({
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
            "lime_url": "/" + result_path,
            "analysis": [
                {"icon": "bi-cpu", "text": "AI mengisolasi fitur wajah kunci menggunakan LIME."},
                {"icon": "bi-eye", "text": "Area yang terlihat adalah bagian yang paling mencurigakan."},
                {"icon": "bi-search", "text": "Periksa distorsi pada area yang tidak tertutup hitam."}
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)