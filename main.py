from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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

# CATALOGUE COMPLET DE TES CULTURES DU TCHAD
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

@app.get("/")
def read_root():
    return {"status": "AgriYedji API en ligne et fonctionnelle"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Lecture sécurisée de la photo avec Pillow
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # 2. Extraction d'une valeur mathématique simple sur les pixels (Simule l'analyse)
        # On récupère les coordonnées de luminosité moyenne
        pixels = list(img.resize((10, 10)).getdata())
        pixel_sum = sum([sum(p) for p in pixels])
        
        # 3. Mapping sur le catalogue tchadien
        highest_pred_index = pixel_sum % len(LABELS)
        
        # Confiance stable simulée
        confidence = 0.85 + ((pixel_sum % 13) / 100.0)

        return {
            "label": LABELS[highest_pred_index],
            "score": float(confidence)
        }
    except Exception as e:
        return {"label": f"Erreur diagnostic : {str(e)}", "score": 0.0}
