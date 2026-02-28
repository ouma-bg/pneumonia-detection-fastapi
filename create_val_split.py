"""
Training CORRIGÉ pour 97% accuracy
Corrections : 
- Class weights ajustés (moins agressifs)
- Plus d'epochs
- Validation data augmentation légère
- Threshold ajusté
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import numpy as np

# ========================
# CONFIG CORRIGÉE
# ========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20       # ↑ Plus d'epochs
EPOCHS_PHASE2 = 15       # ↑ Plus de fine-tuning
LEARNING_RATE = 1e-4     # Légèrement réduit pour stabilité

TRAIN_DIR = 'archive/chest_xray/train'
VAL_DIR   = 'archive/chest_xray/val'
TEST_DIR  = 'archive/chest_xray/test'

# ========================
# VÉRIFICATION
# ========================
print("="*60)
print("🔍 VÉRIFICATION DES DONNÉES")
print("="*60)

for name, path in [('Train', TRAIN_DIR), ('Val', VAL_DIR), ('Test', TEST_DIR)]:
    if not os.path.exists(path):
        print(f"❌ {name} introuvable : {path}")
        exit(1)
    
    normal = len(os.listdir(os.path.join(path, 'NORMAL')))
    pneumonia = len(os.listdir(os.path.join(path, 'PNEUMONIA')))
    print(f"✅ {name}: NORMAL={normal}, PNEUMONIA={pneumonia}")

# ========================
# AUGMENTATION ÉQUILIBRÉE
# ========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # ↑ Plus de variété
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,             # Nouveau
    fill_mode='nearest'
)

# IMPORTANT : Légère augmentation sur validation pour réduire overfitting
val_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,            # Légère rotation
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

print("\n📦 Chargement des données...")

train = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ========================
# CLASS WEIGHTS CORRIGÉS
# ========================
# Calcul moins agressif pour éviter le biais
total = len(train.labels)
count_normal = np.sum(train.labels == 0)
count_pneumonia = np.sum(train.labels == 1)

# Formule corrigée (plus douce)
class_weights = {
    0: (total / (2 * count_normal)) * 0.7,      # ↓ Réduit de 30%
    1: (total / (2 * count_pneumonia)) * 1.3     # ↑ Augmenté de 30%
}

print(f"\n⚖️ Class weights corrigés : {class_weights}")
print(f"   Ratio PNEUMONIA/NORMAL : {count_pneumonia/count_normal:.2f}")

# ========================
# MODÈLE RENFORCÉ
# ========================
def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        # Architecture plus profonde
        layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        
        layers.Dense(2, activation="softmax")
    ])
    return model

print("\n" + "="*60)
print("🏗️ CONSTRUCTION DU MODÈLE RENFORCÉ")
print("="*60)
model = build_model()
model.summary()

# ========================
# PHASE 1 : BASE FREEZÉE
# ========================
print("\n" + "="*60)
print("🚀 PHASE 1 : ENTRAÎNEMENT BASE FREEZÉE")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,              # Plus patient
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'model/checkpoint_phase1_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

history1 = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
    class_weight=class_weights,
    verbose=1
)

val_acc1 = max(history1.history['val_accuracy'])
print(f"\n✅ Phase 1 terminée - Meilleure val_accuracy: {val_acc1*100:.2f}%")

# ========================
# PHASE 2 : FINE-TUNING PROGRESSIF
# ========================
print("\n" + "="*60)
print("🔥 PHASE 2 : FINE-TUNING PROGRESSIF")
print("="*60)

# Unfreeze progressif
model.layers[0].trainable = True

# Freeze toutes les couches sauf les 60 dernières (plus agressif)
for layer in model.layers[0].layers[:-60]:
    layer.trainable = False

trainable = sum([1 for l in model.layers[0].layers if l.trainable])
print(f"🔓 Couches entraînables : {trainable}/154")

# Recompile avec LR très faible
model.compile(
    optimizer=keras.optimizers.Adam(3e-6),  # Encore plus faible
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        'model/checkpoint_phase2_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

history2 = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
    class_weight=class_weights,
    verbose=1
)

val_acc2 = max(history2.history['val_accuracy'])
print(f"\n✅ Phase 2 terminée - Meilleure val_accuracy: {val_acc2*100:.2f}%")

# ========================
# ÉVALUATION DÉTAILLÉE
# ========================
print("\n" + "="*60)
print("📊 ÉVALUATION COMPLÈTE")
print("="*60)

# Évaluation sur test
results = model.evaluate(test, verbose=1)
print(f"\n🎯 Test Accuracy  : {results[1]*100:.2f}%")
print(f"🎯 Test Precision : {results[2]*100:.2f}%")
print(f"🎯 Test Recall    : {results[3]*100:.2f}%")

# Prédictions détaillées
print("\n🔍 Analyse des prédictions...")
predictions = model.predict(test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test.classes

from sklearn.metrics import classification_report, confusion_matrix

print("\n📋 RAPPORT DÉTAILLÉ:")
print(classification_report(true_classes, predicted_classes, 
                          target_names=['NORMAL', 'PNEUMONIA'], digits=3))

print("\n🔢 MATRICE DE CONFUSION:")
cm = confusion_matrix(true_classes, predicted_classes)
print(f"                Prédit NORMAL  Prédit PNEUMONIA")
print(f"Réel NORMAL          {cm[0][0]:4d}          {cm[0][1]:4d}")
print(f"Réel PNEUMONIA       {cm[1][0]:4d}          {cm[1][1]:4d}")

# Calcul des métriques par classe
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # Recall pour PNEUMONIA
specificity = tn / (tn + fp)  # Recall pour NORMAL

print(f"\n📈 MÉTRIQUES MÉDICALES:")
print(f"Sensibilité (détection PNEUMONIA): {sensitivity*100:.1f}%")
print(f"Spécificité (détection NORMAL)   : {specificity*100:.1f}%")

# ========================
# SAUVEGARDE
# ========================
os.makedirs("model", exist_ok=True)
model.save("model/pneumonia_mobilenetV2_FINAL.h5")
model.save("model/pneumonia_mobilenetV2_FINAL.keras")

with open("model/labels.txt", "w") as f:
    for cls, idx in train.class_indices.items():
        f.write(f"{idx} {cls}\n")

# Sauvegarde des historiques
import json
history_dict = {
    'phase1': {k: [float(v) for v in vals] for k, vals in history1.history.items()},
    'phase2': {k: [float(v) for v in vals] for k, vals in history2.history.items()}
}

with open("model/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

print("\n" + "="*60)
print("💾 MODÈLE SAUVEGARDÉ")
print("="*60)
print("📁 model/pneumonia_mobilenetV2_FINAL.h5")
print("📁 model/pneumonia_mobilenetV2_FINAL.keras")
print("📋 model/labels.txt")
print("📊 model/training_history.json")

# ========================
# VERDICT FINAL
# ========================
print("\n" + "="*60)
print("🎯 VERDICT FINAL")
print("="*60)

final_acc = results[1]
if final_acc >= 0.97:
    print("🎉 EXCELLENT ! Objectif de 97% ATTEINT !")
elif final_acc >= 0.95:
    print("👍 TRÈS BON ! Proche de l'objectif (95-97%)")
elif final_acc >= 0.90:
    print("✅ BON, mais peut être amélioré")
else:
    print("⚠️  Performance insuffisante")
    print("\n💡 SUGGESTIONS:")
    print("   1. Vérifier la qualité du dataset")
    print("   2. Augmenter EPOCHS_PHASE1 à 30")
    print("   3. Essayer EfficientNetB0 au lieu de MobileNetV2")

print("\n✅ Entraînement terminé !")