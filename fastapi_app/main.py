from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from pathlib import Path
import logging

from .config import settings
from .models import PredictionResponse, HealthResponse, ErrorResponse
from .utils import PneumoniaDetector, logger

# Créer le dossier uploads
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Initialiser FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le détecteur
detector: PneumoniaDetector = None

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage - Charge le modèle"""
    global detector
    try:
        logger.info("🚀 Démarrage de l'application...")
        detector = PneumoniaDetector(
            model_path=settings.MODEL_PATH,
            labels_path=settings.LABELS_PATH,
            img_size=settings.IMG_SIZE
        )
        logger.info("✅ Application démarrée avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique au démarrage: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("🛑 Arrêt de l'application...")

@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée principal"""
    return {
        "message": f"Bienvenue sur {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "prediction": "/predict"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API
    
    Returns:
        État de l'API et du modèle
    """
    is_healthy = detector is not None and detector.is_loaded()
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        version=settings.VERSION,
        model_path=settings.MODEL_PATH
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        200: {"description": "Prédiction réussie"},
        400: {"description": "Requête invalide"},
        413: {"description": "Fichier trop volumineux"},
        500: {"description": "Erreur serveur"},
        503: {"description": "Service non disponible"}
    }
)
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Détecte la présence de pneumonie sur une radiographie thoracique
    
    **Paramètres:**
    - **file**: Image de radiographie (JPG, JPEG, PNG)
    - Taille maximale: 10 MB
    
    **Retourne:**
    - Prédiction (NORMAL ou PNEUMONIA)
    - Niveau de confiance (0-1)
    - Probabilités pour chaque classe
    - Message d'interprétation
    
    **Exemple de réponse:**
    ```json
    {
        "success": true,
        "prediction": "PNEUMONIA",
        "confidence": 0.9234,
        "probabilities": {
            "PNEUMONIA": 0.9234,
            "NORMAL": 0.0766
        },
        "message": "Confiance très élevée"
    }
    ```
    """
    
    # Vérifier que le modèle est chargé
    if detector is None or not detector.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé. Veuillez réessayer."
        )
    
    # Vérifier le type de fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier doit être une image (JPG, JPEG, PNG)"
        )
    
    # Vérifier l'extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extension non autorisée. Utilisez: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Lire le fichier
        contents = await file.read()
        
        # Vérifier la taille
        if len(contents) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image trop volumineuse (max {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB)"
            )
        
        # Convertir en image PIL
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible de lire l'image: {str(e)}"
            )
        
        # Faire la prédiction
        try:
            result = detector.predict(image)
        except Exception as e:
            logger.error(f"Erreur de prédiction: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors de l'analyse: {str(e)}"
            )
        
        # Retourner la réponse
        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            message=result["message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue: {str(e)}"
        )

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )