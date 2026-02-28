import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Tuple, List, Dict
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PneumoniaDetector:
    """Classe pour gérer la détection de pneumonie"""
    
    def __init__(self, model_path: str, labels_path: str, img_size: int = 224):
        """
        Initialise le détecteur
        
        Args:
            model_path: Chemin vers le fichier .h5 du modèle
            labels_path: Chemin vers le fichier labels.txt
            img_size: Taille des images pour le modèle (224x224)
        """
        self.img_size = img_size
        self.model = None
        self.class_names = []
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Charger le modèle
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Charge le modèle TensorFlow"""
        try:
            logger.info(f"📥 Chargement du modèle depuis: {self.model_path}")
            
            # Vérifier si le fichier existe
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            
            # Clear session et charger
            tf.keras.backend.clear_session()
            self.model = load_model(self.model_path, compile=False)
            
            # Recompiler le modèle
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {str(e)}")
    
    def _load_labels(self):
        """Charge les labels depuis labels.txt"""
        try:
            logger.info(f"📥 Chargement des labels depuis: {self.labels_path}")
            
            if not Path(self.labels_path).exists():
                raise FileNotFoundError(f"Fichier labels non trouvé: {self.labels_path}")
            
            with open(self.labels_path, 'r') as f:
                # Format: "0 PNEUMONIA\n1 NORMAL\n"
                self.class_names = [line.strip().split(" ")[1] for line in f.readlines()]
            
            logger.info(f"✅ Labels chargés: {self.class_names}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des labels: {e}")
            raise RuntimeError(f"Impossible de charger les labels: {str(e)}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Prétraite l'image pour le modèle
        
        Args:
            image: Image PIL
            
        Returns:
            Array numpy normalisé (1, 224, 224, 3)
        """
        try:
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionner
            image = image.resize((self.img_size, self.img_size))
            
            # Convertir en array et normaliser
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Ajouter la dimension batch
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du prétraitement: {e}")
            raise ValueError(f"Erreur de prétraitement de l'image: {str(e)}")
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Effectue une prédiction sur l'image
        
        Args:
            image: Image PIL
            
        Returns:
            Dictionnaire avec la prédiction et les probabilités
        """
        try:
            # Prétraiter l'image
            img_array = self.preprocess_image(image)
            
            # Prédiction
            logger.info("🔍 Analyse en cours...")
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Obtenir la classe prédite
            class_idx = np.argmax(predictions)
            predicted_class = self.class_names[class_idx]
            confidence = float(predictions[class_idx])
            
            # Créer le dictionnaire de probabilités
            probabilities = {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions)
            }
            
            # Message selon la confiance
            if confidence > 0.95:
                message = "Confiance très élevée - Le modèle est très sûr de sa prédiction"
            elif confidence > 0.85:
                message = "Bonne confiance - Le modèle est relativement sûr"
            elif confidence > 0.70:
                message = "Confiance modérée - Il est recommandé de consulter un médecin"
            else:
                message = "Faible confiance - Résultat incertain, consultation médicale nécessaire"
            
            logger.info(f"✅ Prédiction: {predicted_class} ({confidence:.2%})")
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            raise RuntimeError(f"Erreur lors de la prédiction: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self.model is not None and len(self.class_names) > 0 