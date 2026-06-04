from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PIL import Image
import io
import os
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
# Le serveur va lire la clé de manière cachée et sécurisée
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def read_root():
    return {"status": "AgriYedji Engine Expert en ligne"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 2. Lecture de l'image envoyée par Flutter
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # 3. Rédaction du prompt d'expertise agronomique stricte
        prompt = (
            "Agis en tant qu'expert agronome AgriYedji au Tchad.\n"
            "Analyse cette photo de plante et génère une réponse structurée au format JSON strict avec trois clés :\n\n"
            "1. 'titre': Nom de la plante suivi de sa maladie en français (ex: 'Manguier - Oïdium')."
            "Si la plante est saine, écris 'Nom de la plante - Sain'.\n"
            "2. 'symptomes': Décris brièvement en une phrase ce qui se passe sur la photo (ex: 'Présence d'un feutrage blanc poudreux sur la surface des feuilles').\n"
            "3. 'solution': Donne un conseil de traitement clair, biologique/accessible au Tchad, et explique comment l'appliquer.\n\n"
            "Ne renvoie rien d'autre que le dictionnaire JSON."
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
            "titre": data.get("titre", "Culture - Diagnostic inconnu"),
            "symptomes": data.get("symptomes", "Symptômes non déterminés."),
            "solution": data.get("solution", "Veuillez contacter un conseiller agricole."),
            "score": 0.96
        }

    except Exception as e:
        return {
            "titre": "Analyse impossible",
            "symptomes": "Erreur technique lors de la lecture de l'image.",
            "solution": f"Détails : {str(e)}",
            "score": 0.0
        }
