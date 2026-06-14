"""
LukOut - prepoznavanje jestivih i nejestivih biljaka iz familije luka
Autorke: Nina Antic IN25/2023, Tamara Sevo IN27/2023
Arhitektura: nasa CNN "od nule" 
Klase: edible=0, inedible=1
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
import keras
from keras import layers

from lukout_izvestaj import print_report, plot_confusion_matrix, full_eval, print_final_report

# Konfiguracija
IMG_SIZE   = (224, 224)
BATCH_SIZE = 64
SEED       = 42

EPOCHS = 60
LR     = 1e-4

# ispod ovog praga -> nejestivo 
CONFIDENCE_THRESHOLDS = [0.5, 0.9]

DATASET_DIR = Path("dataset_prepared")
RESULTS_DIR = Path("results_nasaNN")
RESULTS_DIR.mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Ucitavanje podataka
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

test_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

CLASS_NAMES  = train_ds.class_names  # edible=0, inedible=1
EDIBLE_IDX   = CLASS_NAMES.index("edible")
INEDIBLE_IDX = CLASS_NAMES.index("inedible")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Gradnja modela
inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 255)(inputs)  # EfficientNet je ovo radio interno, sada moramo mi

x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu",
                  kernel_regularizer=keras.regularizers.L2(1e-4))(x)
x = layers.Dropout(0.50)(x)
x = layers.Dense(32, activation="relu",
                  kernel_regularizer=keras.regularizers.L2(1e-4))(x)
x = layers.Dropout(0.30)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="LukOut_nasaNN")

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)
model.summary()

# Trening 
cb = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15,
        restore_best_weights=True, verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, verbose=1,
    ),
]

hist = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=cb,
)
history = hist.history

# Evaluacija 
def predict_and_collect(ds):
    all_raw = []
    all_true = []
    for x_batch, y_batch in ds:
        raw_batch = model(x_batch, training=False).numpy().flatten()
        all_raw.append(raw_batch)
        all_true.append(y_batch.numpy().flatten())

    raw_probs = np.concatenate(all_raw)
    true_lbls = np.concatenate(all_true).astype(int)
    p_edible  = 1.0 - raw_probs
    return true_lbls, p_edible


true_val,  p_edible_val  = predict_and_collect(val_ds)
true_test, p_edible_test = predict_and_collect(test_ds)

# Vizualizacije
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# Krive treninga
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("LukOut nasaNN - Istorija treninga", fontsize=14, fontweight="bold")

metric_triples = [
    ("accuracy",  "val_accuracy",  "Tacnost (Accuracy)"),
    ("loss",      "val_loss",      "Gubitak (Loss)"),
    ("auc",       "val_auc",       "AUC-ROC"),
    ("precision", "val_precision", "Preciznost (Precision)"),
]

for ax, (tk, vk, title) in zip(axes.flatten(), metric_triples):
    ep = range(1, len(history[tk]) + 1)
    ax.plot(ep, history[tk], label="Trening",    color="steelblue",  lw=2)
    ax.plot(ep, history[vk], label="Validacija", color="darkorange", lw=2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "01_training_history.png", bbox_inches="tight")
plt.show()

# Evaluacija i izvestaj za svaki safety prag
for threshold in CONFIDENCE_THRESHOLDS:
    print(f"\n{'#'*60}\n  PRAG = {threshold}\n{'#'*60}")

    out_dir = RESULTS_DIR / f"prag_{threshold}"
    out_dir.mkdir(exist_ok=True)

    pred_val  = np.where(p_edible_val  >= threshold, EDIBLE_IDX, INEDIBLE_IDX)
    pred_test = np.where(p_edible_test >= threshold, EDIBLE_IDX, INEDIBLE_IDX)

    cm_val,  fpr_val,  tpr_val,  auc_val  = full_eval(true_val,  pred_val,  p_edible_val,  "VALIDACIJA", CLASS_NAMES, EDIBLE_IDX, threshold)
    cm_test, fpr_test, tpr_test, auc_test = full_eval(true_test, pred_test, p_edible_test, "TEST", CLASS_NAMES, EDIBLE_IDX, threshold)

    # Matrice konfuzije
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Matrice konfuzije (prag = {threshold})", fontsize=13, fontweight="bold")
    plot_confusion_matrix(axes[0], cm_val,  CLASS_NAMES, "Validacioni skup")
    plot_confusion_matrix(axes[1], cm_test, CLASS_NAMES, "Test skup")
    plt.tight_layout()
    plt.savefig(out_dir / "02_confusion_matrices.png", bbox_inches="tight")
    plt.show()

    # ROC krive
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_val,  tpr_val,  lw=2, color="darkorange",
            label=f"Validacija (AUC = {auc_val:.3f})")
    ax.plot(fpr_test, tpr_test, lw=2, color="steelblue",
            label=f"Test       (AUC = {auc_test:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Slucajni klasifikator")
    ax.set_xlabel("Stopa lazno pozitivnih (FPR)")
    ax.set_ylabel("Stopa tacno pozitivnih (TPR)")
    ax.set_title(f"ROC kriva - LukOut nasaNN (prag = {threshold})", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_roc_curve.png", bbox_inches="tight")
    plt.show()

    # Finalni izvestaj
    print_final_report(true_val, pred_val, true_test, pred_test,
                        auc_val, auc_test, CLASS_NAMES, INEDIBLE_IDX,
                        threshold,
                        "CNN od nule (bez transfer learninga)", "LukOut nasaNN")
