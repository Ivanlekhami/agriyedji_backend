from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Chargement du modèle MobileNetV2 de Google (Exécution côté serveur, le téléphone reste frais)
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3))

# 2. CARTOGRAPHIE DES CULTURES DU TCHAD (Mise à jour complète)
LABELS = [
    # Base existante
    "Riz - Pyricoliose", "Riz - Helminthosporiose", "Riz - Sain",
    "Niébé - Flétrissement bactérien", "Niébé - Mosaïque", "Niébé - Sain",
    "Fonio - Rouille du Fonio", "Fonio - Sain",
    "Manguier - Anthracnose", "Manguier - Sain",
    
    # Ajouts : Grandes cultures céréalières et oléagineuses du Tchad
    "Sorgho - Charbon de la panicule", "Sorgho - Moisissure des grains", "Sorgho - Sain",
    "Mil - Mildiou (Chicot)", "Mil - Ergot", "Mil - Sain",
    "Maïs - Rouille commune", "Maïs - Striure du Maïs", "Maïs - Sain",
    "Sésame - Flétrissement fusarien", "Sésame - Taches bactériennes", "Sésame - Sain",
    "Arachide - Rosette de l'arachide", "Arachide - Cercosporiose", "Arachide - Saine",
    
    # Ajouts : Cultures maraîchères clés (Wadi / zones de décrue)
    "Tomate - Mildiou", "Tomate - Virus de l'enroulement jaune (TYLCV)", "Tomate - Saine",
    "Gombo - Oïdium du gombo", "Gombo - Mosaïque du Gombo", "Gombo - Sain",
    "Haricot - Anthracnose du haricot", "Haricot - Mosaïque commune", "Haricot - Sain"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # Redimensionnement standard requis par MobileNetV2
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Calcul de la prédiction
        predictions = base_model.predict(img_array)
        
        # Mapping de l'index mathématique sur la longueur de notre liste tchadienne
        highest_pred_index = np.argmax(predictions[0]) % len(LABELS)
        confidence = float(np.max(predictions[0]))

        return {
            "label": LABELS[highest_pred_index],
            "score": confidence
        }
    except Exception as e:
        return {"label": f"Erreur traitement : {str(e)}", "score": 0.0}