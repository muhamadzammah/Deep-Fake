import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_PATH = "dataset"
MODEL_PATH = "best_model.keras"

# =========================
# LOAD DATA
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
# SPLIT
# =========================
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

y_prob = model.predict(val_ds).ravel()
y_pred = (y_prob > 0.4).astype(int)

y_true = np.concatenate([y for _, y in val_ds], axis=0)

# =========================
# METRICS
# =========================
cm = confusion_matrix(y_true, y_pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

report = classification_report(y_true, y_pred)

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
plt.show()

# =========================
# PRECISION - RECALL
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.savefig("precision_recall_curve.png")
plt.show()

# =========================
# CONFUSION MATRIX HEATMAP
# =========================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# METRICS BAR CHART
# =========================
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
metrics_values = [acc, prec, rec, f1]

plt.figure()
plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.grid(axis="y")

for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.savefig("metrics_bar.png")
plt.show()

# =========================
# OUTPUT TE TERMINAL
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
print(f"AUC       : {roc_auc:.4f}")

# =========================
# SIMPAN FILE
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
    f.write(f"AUC       : {roc_auc:.4f}\n")

print("\n✅ Semua hasil + visualisasi berhasil disimpan!")