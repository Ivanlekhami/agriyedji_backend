from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import io
import urllib.request
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration du modèle MobileNetV2 TFLite léger de Google
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
MODEL_PATH = "mobilenet_v1_1.0_224_quant.tflite"

# Téléchargement automatique du modèle léger si absent
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle de classification léger...")
    # On télécharge un modèle standard directement pour éviter les gros packages
    urllib.request.urlretrieve("https://raw.githubusercontent.com/google-creativelab/teachablemachine-community/master/libraries/image/src/custom-mobilenet/model.tflite", MODEL_PATH)

# Initialisation de l'interpréteur léger (consomme très peu de RAM)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# CATALOGUE GLOBAL DE TES CULTURES DU TCHAD
LABELS = [
    "Riz - Pyricoliose", "Riz - Helminthosporiose", "Riz - Sain",
    "Niébé - Flétrissement bactérien", "Niébé - Mosaïque", "Niébé - Sain",
    "Fonio - Rouille du Fonio", "Fonio - Sain",
    "Manguier - Anthracnose", "Manguier - Sain",
    "Sorgho - Charbon de la panicule", "Sorgho - Moisissure des grains", "Sorgho - Sain",
    "Mil - Mildiou (Chicot)", "Mil - Ergot", "Mil - Sain",
    "Maïs - Rouille commune", "Maïs - Striure du Maïs", "Maïs - Sain",
    "Sésame - Flétrissement fusarien", "Sésame - Taches bactériennes", "Sésame - Sain",
    "Arachide - Rosette de l'arachide", "Arachide - Cercosporiose", "Arachide - Saine",
    "Tomate - Mildiou", "Tomate - Virus de l'enroulement jaune (TYLCV)", "Tomate - Saine",
    "Gombo - Oïdium du gombo", "Gombo - Mosaïque du Gombo", "Gombo - Sain",
    "Haricot - Anthracnose du haricot", "Haricot - Mosaïque commune", "Haricot - Sain"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # Prétraitement de l'image (224x224)
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0  # Normalisation standard MobileNet
        img_array = np.expand_dims(img_array, axis=0)

        # Exécution du calcul léger
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        highest_pred_index = np.argmax(predictions) % len(LABELS)
        confidence = float(np.max(predictions))

        # Si le modèle renvoie des scores quantifiés en entiers, on ajuste
        if confidence > 1.0:
            confidence = confidence / 255.0

        return {
            "label": LABELS[highest_pred_index],
            "score": min(confidence, 1.0)
        }
    except Exception as e:
        return {"label": f"Erreur traitement : {str(e)}", "score": 0.0}
