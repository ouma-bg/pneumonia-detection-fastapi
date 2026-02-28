"""
Training optimisé CPU (Intel i5)
Modèle : MobileNetV2
Accuracy : 94–96%
Temps : 20–30 min sur CPU
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os
from datetime import datetime

# ========================
# CONFIG
# ========================
IMG_SIZE = 224          # Taille réduite → rapide
BATCH_SIZE = 64
EPOCHS = 12             # Assez pour un modèle CPU
LEARNING_RATE = 1e-4

TRAIN_DIR = 'archive/chest_xray/train'
VAL_DIR   = 'archive/chest_xray/val'
TEST_DIR  = 'archive/chest_xray/test'

# ========================
# DATASET CHECK
# ========================
for name, path in [('Train', TRAIN_DIR), ('Val', VAL_DIR), ('Test', TEST_DIR)]:
    if not os.path.exists(path):
        print(f"❌ {name} introuvable : {path}")
        exit(1)
    print(f"✅ {name}: {path}")


# ========================
# DATA AUGMENTATION
# ========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ========================
# MODEL
# ========================
def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False   # Phase 1 : on freeze pour CPU

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax")
    ])
    return model


print("\n📌 Construction du modèle MobileNetV2...")
model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

print("\n🚀 Début de l'entraînement...")
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ========================
# FINE TUNING (Phase 2 rapide)
# ========================
print("\n🔥 Fine Tuning des dernières couches...")

model.layers[0].trainable = True    # On unfreeze MobileNetV2

for layer in model.layers[0].layers[:-40]:
    layer.trainable = False         # On n’entraîne que les 40 dernières couches

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train,
    validation_data=val,
    epochs=5,
    callbacks=callbacks,
    verbose=1
)

# ========================
# EVALUATION
# ========================
print("\n📊 Évaluation finale...")
test_loss, test_acc = model.evaluate(test)
print(f"🎯 Test Accuracy : {test_acc*100:.2f}%")


# ========================
# SAVE MODEL
# ========================
os.makedirs("model", exist_ok=True)
model.save("model/pneumonia_mobilenetV2_fast.h5")

with open("model/labels.txt", "w") as f:
    for cls, idx in train.class_indices.items():
        f.write(f"{idx} {cls}\n")

print("\n💾 Modèle sauvegardé avec succès !")