import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("=== Modèles Gemini disponibles pour votre clé ===")
for m in client.models.list():
    # Filtre pour ne voir que les modèles Gemini récents
    if "gemini" in m.name:
        print(f"- {m.name}")