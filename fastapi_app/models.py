from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions"""
    success: bool
    prediction: str = Field(..., description="Classe prédite (NORMAL ou PNEUMONIA)")
    confidence: float = Field(..., ge=0, le=1, description="Niveau de confiance (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilités pour chaque classe")
    message: str = Field(default="", description="Message d'interprétation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": "PNEUMONIA",
                "confidence": 0.9234,
                "probabilities": {
                    "PNEUMONIA": 0.9234,
                    "NORMAL": 0.0766
                },
                "message": "Confiance très élevée - Le modèle est très sûr de sa prédiction"
            }
        }

class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str = Field(..., description="État de l'API (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Le modèle est-il chargé ?")
    version: str = Field(..., description="Version de l'API")
    model_path: str = Field(..., description="Chemin du modèle")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "model_path": "../model/pneumonia_mobilenetV2_optimized.h5"
            }
        }

class ErrorResponse(BaseModel):
    """Modèle de réponse pour les erreurs"""
    success: bool = False
    error: str
    detail: str = ""