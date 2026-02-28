"""Script de test pour l'API"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test du endpoint health"""
    print("\n🔍 Test du health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Réponse: {response.json()}")
    return response.status_code == 200

def test_predict(image_path: str):
    """Test du endpoint predict"""
    print(f"\n🔍 Test de prédiction avec: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Fichier non trouvé: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Résultat:")
        print(f"  - Prédiction: {result['prediction']}")
        print(f"  - Confiance: {result['confidence']:.2%}")
        print(f"  - Message: {result['message']}")
        print(f"  - Probabilités:")
        for classe, prob in result['probabilities'].items():
            print(f"    • {classe}: {prob:.2%}")
        return True
    else:
        print(f"❌ Erreur: {response.json()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DE L'API PNEUMONIA DETECTION")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ Health check échoué!")
        sys.exit(1)
    
    # Test prédiction si une image est fournie
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_predict(image_path)
    else:
        print("\n💡 Usage: python test_api.py <chemin_image>")
        print("Exemple: python test_api.py ../archive/chest_xray/test/NORMAL/NORMAL-1.jpeg")
    
    print("\n" + "=" * 60)