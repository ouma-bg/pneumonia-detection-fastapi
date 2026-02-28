import tensorflow as tf
from tensorflow.keras.models import load_model

def load_model_with_fix(path):
    """
    Charge le modèle de manière sécurisée avec gestion des erreurs
    Compatible avec les formats .h5 et .keras
    """
    # Reset du contexte Keras/TensorFlow
    tf.keras.backend.clear_session()
    
    try:
        # Tentative de chargement sans compilation
        model = load_model(path, compile=False)
        
        # Recompilation avec les mêmes paramètres que l'entraînement
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        print(f"✅ Modèle chargé avec succès : {path}")
        return model
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        
        # Tentative avec format alternatif
        if path.endswith('.h5'):
            alt_path = path.replace('.h5', '.keras')
        else:
            alt_path = path.replace('.keras', '.h5')
            
        try:
            print(f"🔄 Tentative avec format alternatif : {alt_path}")
            model = load_model(alt_path, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            print(f"✅ Modèle chargé avec format alternatif")
            return model
        except:
            raise Exception(f"Impossible de charger le modèle depuis {path} ou {alt_path}")