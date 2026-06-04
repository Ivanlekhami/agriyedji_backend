from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PIL import Image
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialisation du client Gemini
# Remplace par ta vraie clé API obtenue sur Google AI Studio
GEMINI_API_KEY = "AIzaSyBEDvr4Nyjda972se9LmIsh0n-JVe6aOiI"
client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def read_root():
    return {"status": "AgriYedji Engine avec Gemini est en ligne"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 2. Lecture de l'image envoyée par Flutter
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # 3. Rédaction du prompt d'expertise agronomique stricte
        prompt = (
            "Agis en tant qu'expert agronome AgriYedji au Tchad. Analyse cette photo de plante.\n"
            "Tu dois impérativement répondre sous un format JSON strict contenant deux clés :\n"
            "1. 'label': Le nom de la plante suivi de sa maladie en français (ex: 'Manguier - Anthracnose' ou 'Riz - Pyricoliose'). "
            "Si la plante est saine, écris 'Nom de la plante - Sain'.\n"
            "2. 'conseil': Un conseil de traitement court, biologique ou accessible, adapté au contexte tchadien.\n"
            "Ne donne aucune explication en dehors du JSON."
        )

        # 4. Appel de Gemini depuis le serveur Render (Zéro blocage opérateur)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt]
        )

        # 5. Extraction et nettoyage de la réponse JSON de Gemini
        response_text = response.text.strip()
        # Sécurité si Gemini ajoute des balises ```json
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
        data = json.loads(response_text)

        return {
            "label": data.get("label", "Plante - Maladie inconnue"),
            "score": 0.95, # Score de confiance élevé simulé pour l'interface Flutter
            "conseil": data.get("conseil", "Consultez un conseiller agricole.")
        }

    except Exception as e:
        return {
            "label": "Erreur - Analyse impossible",
            "score": 0.0,
            "conseil": f"Détails de l'erreur : {str(e)}"
        }
