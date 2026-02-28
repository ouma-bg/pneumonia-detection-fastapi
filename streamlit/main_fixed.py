import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

# ========================
# CONFIGURATION
# ========================
st.set_page_config(
    page_title="Pneumonia AI Detector",
    page_icon="🩻",
    layout="centered"
)

IMG_SIZE = 224
MODEL_PATH = "../model/pneumonia_mobilenetV2_optimized.h5"
LABELS_PATH = "../model/labels.txt"
BG_IMAGE_PATH = "./bgs/bg5.png"

# ========================
# STYLE ET FOND D'ÉCRAN
# ========================
def set_background(image_path):
    """Définir l'image de fond"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            /* Fond d'écran principal */
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: contain;
                background-position: right top;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Container principal avec fond blanc semi-transparent */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.75);
                padding: 2rem;
                border-radius: 15px;
                max-width: 900px;
                backdrop-filter: blur(5px);
            }}
            
            /* Titre principal */
            h1 {{
                background: linear-gradient(135deg, rgba(255, 193, 7, 0.95), rgba(255, 152, 0, 0.95));
                color: white !important;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }}
            
            /* Sous-titres */
            h2, h3 {{
                color: #1976d2 !important;
                margin-top: 15px;
            }}
            
            /* File uploader styling */
            [data-testid="stFileUploader"] {{
                background-color: rgba(33, 37, 41, 0.95);
                padding: 20px;
                border-radius: 10px;
                border: 2px dashed #ffc107;
            }}
            
            [data-testid="stFileUploader"] label {{
                color: #ffc107 !important;
                font-weight: bold;
            }}
            
            /* Columns */
            [data-testid="column"] {{
                background-color: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            /* Success et Error boxes */
            .stSuccess, .stError, .stWarning, .stInfo {{
                background-color: rgba(255, 255, 255, 0.95) !important;
                border-radius: 10px;
                padding: 15px;
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                background-color: rgba(255, 193, 7, 0.2);
                border-radius: 8px;
                font-weight: bold;
            }}
            
            /* Divider */
            hr {{
                margin: 20px 0;
                border-color: rgba(255, 193, 7, 0.5);
            }}
            
            /* Spinner */
            .stSpinner > div {{
                border-top-color: #ffc107 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"⚠️ Image de fond non trouvée : {image_path}")

# Appliquer le fond d'écran
set_background(BG_IMAGE_PATH)

# ========================
# FONCTIONS UTILITAIRES
# ========================
@st.cache_resource
def load_model_cached():
    """Charge le modèle une seule fois"""
    tf.keras.backend.clear_session()
    model = load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

@st.cache_data
def load_labels():
    """Charge les labels"""
    with open(LABELS_PATH) as f:
        return [x.split(" ")[1].strip() for x in f.readlines()]

def preprocess_image(img: Image.Image):
    """Prétraitement de l'image"""
    # Conversion en RGB si nécessaire
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify(image, model, class_names):
    """Classification de l'image"""
    img = preprocess_image(image)
    preds = model.predict(img, verbose=0)[0]
    
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    
    # ✅ CORRECTION : Conversion explicite en float Python
    return class_names[class_idx], float(confidence), [float(p) for p in preds]

# ========================
# INTERFACE STREAMLIT
# ========================
st.title("🩻 Détection de Pneumonie par IA")

st.divider()

# Chargement du modèle
try:
    with st.spinner("Chargement du modèle IA..."):
        model = load_model_cached()
        class_names = load_labels()
    
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    st.stop()

# Upload d'image
st.subheader("📤 Télécharger une radiographie")
uploaded_file = st.file_uploader(
    "Choisissez une image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Téléchargez une radiographie thoracique de face"
)

if uploaded_file:
    # Affichage de l'image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Image téléchargée")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analyse")
        
        with st.spinner("Analyse en cours..."):
            label, confidence, all_preds = classify(image, model, class_names)
        
        # Affichage du résultat principal
        if label == "PNEUMONIA":
            st.error(f"### ⚠️ PNEUMONIE DÉTECTÉE")
            st.metric("Confiance", f"{confidence*100:.1f}%")
        else:
            st.success(f"### ✅ NORMAL")
            st.metric("Confiance", f"{confidence*100:.1f}%")
        
        # Détails des probabilités
        st.divider()
        st.write("**Probabilités détaillées :**")
        for i, name in enumerate(class_names):
            # ✅ CORRECTION : all_preds[i] est maintenant un float Python
            st.progress(all_preds[i], text=f"{name}: {all_preds[i]*100:.1f}%")
        
        # Interprétation
        st.divider()
        if confidence > 0.95:
            st.info("🔬 **Confiance très élevée** - Le modèle est très sûr de sa prédiction")
        elif confidence > 0.85:
            st.info("🔬 **Bonne confiance** - Le modèle est relativement sûr")
        elif confidence > 0.70:
            st.warning("⚠️ **Confiance modérée** - Il est recommandé de consulter un médecin")
        else:
            st.warning("⚠️ **Faible confiance** - Résultat incertain, consultation médicale nécessaire")

# Informations supplémentaires
st.divider()
with st.expander("ℹ️ À propos du modèle"):
    st.markdown("""
    **Architecture :** MobileNetV2 (Transfer Learning)
    
    **Entraînement :**
    - Dataset : Chest X-Ray Images (Kaggle)
    - Précision : ~97%
    - Images d'entraînement : 5,216
    
    **Avertissement :**
    Ce modèle est un outil d'aide à la décision éducatif. Il ne remplace en aucun cas 
    l'expertise d'un professionnel de santé qualifié. Toujours consulter un médecin 
    pour un diagnostic médical.
    """)

with st.expander("❓ Comment utiliser"):
    st.markdown("""
    1. Téléchargez une radiographie thoracique de face
    2. Attendez l'analyse automatique
    3. Consultez les résultats et la confiance du modèle
    4. En cas de doute, consultez toujours un médecin
    """)