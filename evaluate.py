import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_PATH = "dataset"
MODEL_PATH = "best_model.keras"

# =========================
# LOAD DATA VALIDASI
# =========================
def load_data():
    data, labels = [], []

    for label in ["fake", "real"]:
        folder = os.path.join(DATASET_PATH, "train", label)

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for f in files:
            data.append(os.path.join(folder, f))
            labels.append(0 if label == "fake" else 1)

    return data, np.array(labels)

data, labels = load_data()

# =========================
# SPLIT VALIDASI (SAMA SEPERTI TRAIN)
# =========================
from sklearn.model_selection import train_test_split

_, val_data, _, val_labels = train_test_split(
    data, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("Jumlah data validasi:", len(val_data))

# =========================
# PREPROCESSING
# =========================
def parse_function(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

val_ds = build_dataset(val_data, val_labels)

# =========================
# LOAD MODEL
# =========================
print("\n🔄 Load model terbaik...")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDIKSI
# =========================
print("📊 Melakukan prediksi...")

y_pred = model.predict(val_ds)
y_pred = (y_pred > 0.4).astype(int).flatten()

y_true = np.concatenate([y for _, y in val_ds], axis=0)

# =========================
# EVALUASI
# =========================
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# =========================
# OUTPUT KE TERMINAL
# =========================
print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(report)

print("\n===== METRICS =====")
print(f"Akurasi   : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

# =========================
# SIMPAN KE FILE (UNTUK SKRIPSI)
# =========================
with open("hasil_evaluasi.txt", "w") as f:
    f.write("===== CONFUSION MATRIX =====\n")
    f.write(str(cm) + "\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report + "\n")

    f.write("===== METRICS =====\n")
    f.write(f"Akurasi   : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n")

print("\n✅ Hasil evaluasi disimpan ke: hasil_evaluasi.txt")