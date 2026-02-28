from pathlib import Path

class Settings:
    """Configuration de l'application"""
    
    # Informations de l'application
    APP_NAME: str = "Pneumonia Detection API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API de détection de pneumonie par IA à partir de radiographies thoraciques"
    
    # Chemins des fichiers
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH: str = str(BASE_DIR / "model" / "pneumonia_mobilenetV2_optimized.h5")
    LABELS_PATH: str = str(BASE_DIR / "model" / "labels.txt")
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    
    # Paramètres du modèle
    IMG_SIZE: int = 224
    
    # Limites de fichiers
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png"}
    
    # Configuration serveur
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

settings = Settings()