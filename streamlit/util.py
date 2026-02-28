import numpy as np
from PIL import Image

# ✅ Même taille que l'entraînement
IMG_SIZE = 224

def preprocess_image(img: Image.Image):
    """
    Prétraite une image pour la prédiction
    
    Args:
        img: Image PIL
        
    Returns:
        np.array: Image prétraitée prête pour la prédiction
    """
    # Conversion en RGB si nécessaire (gère les images en niveaux de gris)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionnement à la taille attendue
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Conversion en array et normalisation [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Ajout de la dimension batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def classify(image, model, class_names):
    """
    Classifie une image avec le modèle
    
    Args:
        image: Image PIL
        model: Modèle Keras chargé
        class_names: Liste des noms de classes
        
    Returns:
        tuple: (nom_classe, score_confiance, toutes_probabilités)
    """
    # Prétraitement
    img = preprocess_image(image)
    
    # Prédiction
    predictions = model.predict(img, verbose=0)[0]
    
    # Classe prédite
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    
    return class_names[class_idx], float(confidence), predictions.tolist()


def get_prediction_details(image, model, class_names):
    """
    Retourne des détails complets sur la prédiction
    
    Returns:
        dict: Dictionnaire avec tous les détails
    """
    label, confidence, all_preds = classify(image, model, class_names)
    
    return {
        'predicted_class': label,
        'confidence': confidence,
        'all_probabilities': {
            class_names[i]: all_preds[i] for i in range(len(class_names))
        },
        'is_high_confidence': confidence > 0.90,
        'is_pneumonia': label == 'PNEUMONIA'
    }