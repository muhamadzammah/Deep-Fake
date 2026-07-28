import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2
import numpy as np
import os
import random
import signal
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 🔥 OPTIMASI M1
# =========================
tf.config.optimizer.set_jit(True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_PATH = "dataset"

MAX_PER_CLASS = 10000
VAL_SPLIT = 0.2

EPOCHS = 10
FINE_TUNE_EPOCHS = 10

CHECKPOINT_PATH = "best_model.keras"
RESUME_PATH = "resume_model.keras"

# =========================
# LOAD DATA
# =========================
def load_subset():
    data, labels = [], []

    for label in ["fake", "real"]:
        folder = os.path.join(DATASET_PATH, "train", label)

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        random.shuffle(files)
        files = files[:MAX_PER_CLASS]

        for f in files:
            data.append(os.path.join(folder, f))
            labels.append(0 if label == "fake" else 1)

    return data, np.array(labels)

data, labels = load_subset()

train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels,
    test_size=VAL_SPLIT,
    stratify=labels,
    random_state=42
)

print("Train:", len(train_data))
print("Val:", len(val_data))

# =========================
# PARSE FUNCTION
# =========================
def parse_function(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

# =========================
# DATA PIPELINE 🔥
# =========================
def build_dataset(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.ignore_errors()  # 🔥 versi baru (no warning)

    if training:
        ds = ds.shuffle(2000)

    ds = ds.batch(BATCH_SIZE)

    if training:
        ds = ds.repeat()  # 🔥 biar tidak kehabisan data

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

train_ds = build_dataset(train_data, train_labels, True)
val_ds = build_dataset(val_data, val_labels, False)

# =========================
# AUGMENTATION
# =========================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.1),
    layers.GaussianNoise(0.03),
])

# =========================
# BUILD MODEL
# =========================
def build_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = augmentation(inputs)

    base_model = EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid", dtype='float32')(x)

    model = models.Model(inputs, outputs)

    return model, base_model

# =========================
# RESUME TRAINING 🔥
# =========================
if os.path.exists(RESUME_PATH):
    print("🔄 Resume training dari checkpoint...")
    model = tf.keras.models.load_model(RESUME_PATH)
    base_model = model.layers[2]
else:
    print("🆕 Training dari awal...")
    model, base_model = build_model()

# =========================
# CALLBACKS
# =========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================
# SAVE SAAT CTRL+C 🔥
# =========================
def save_on_exit(signal, frame):
    print("\n⚠️ Training dihentikan manual (CTRL+C)")
    print("💾 Menyimpan model...")
    model.save(RESUME_PATH)
    print("✅ Model tersimpan, bisa dilanjutkan.")
    sys.exit(0)

signal.signal(signal.SIGINT, save_on_exit)

# =========================
# STEP CONFIG 🔥
# =========================
steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(val_data) // BATCH_SIZE

# =========================
# TRAIN AWAL
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# =========================
# FINE TUNING
# =========================
base_model.trainable = True

for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# =========================
# LOAD BEST MODEL
# =========================
print("\n🔄 Load best model...")
model = tf.keras.models.load_model(CHECKPOINT_PATH)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(val_ds)
y_pred = (y_pred > 0.4).astype(int)

y_true = np.concatenate([y for _, y in val_ds], axis=0)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# =========================
# SAVE FINAL
# =========================
model.save("final_model_b2_m1.keras")

print("✅ TRAINING SELESAI (BEST MODEL DIGUNAKAN)")