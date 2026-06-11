import os
import base64
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import magic
from tqdm import tqdm
from pydantic import BaseModel, model_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
# from langchain_community.document_loaders.parsers import LanguageParser
# from .tools import MinimalSource


load_dotenv()


class Chunker(BaseModel):
    data_base: Path
    list_other: list[Path] = []
    list_txt: list[Path] = []
    list_img: list[Path] = []

    @model_validator(mode='after')
    def find_suffix(self) -> 'Chunker':
        for element in tqdm(self.data_base.rglob('*'),
                            desc="Extraction des suffix"):
            if element.is_file():

                mime = magic.Magic(mime=True)
                type_mime = mime.from_file(element)

                s = element.suffix

                if type_mime.startswith('text/'):
                    if s not in self.list_txt:
                        self.list_txt.append(s)

                elif type_mime.startswith('application/'):
                    if s not in self.list_txt:
                        self.list_txt.append(s)

                elif type_mime.startswith('image/'):
                    if s != ".ico":
                        self.list_img.append(element)
        return self

    def _encode_image(self, image_path: Path) -> str:
        """Encode une image en Base64 pour l'envoyer au modèle via l'API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def loader_database(self) -> None:
        print(f"\nTraitement de {len(self.list_img)} images avec Gemini...")

        # Initialisation du modèle (pas besoin de parseur obsolète !)
        llm_vision = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            timeout=30,
            max_retries=1
        )
        docs_images = []

        # Boucle d'analyse manuelle avec barre de progression
        for img_path in tqdm(self.list_img, desc="Analyse Vision"):
            try:
                file_size_kb = os.path.getsize(img_path) / 1024
                if file_size_kb > 4000: # Si l'image fait plus de 4 Mo, on l'ignore (trop lourde pour API)
                    print(f"\n⚠️ Fichier ignoré (trop lourd, {file_size_kb:.0f} Ko): {img_path.name}")
                    continue
                base64_image = self._encode_image(img_path)

                # Format standard LangChain pour l'envoi d'images
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Décris cette image technique en détail. Si c'est un schéma, décris l'architecture. Si c'est du texte ou du code, retranscris-le."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                )

                reponse = llm_vision.invoke([message])

                texte_description = ""

                # Si la réponse est une simple chaîne (le comportement standard attendu)
                if isinstance(reponse.content, str):
                    texte_description = reponse.content

                # Si la réponse est une liste (le comportement de Gemini Vision)
                elif isinstance(reponse.content, list):
                    # On cherche le premier élément qui contient du texte
                    for element in reponse.content:
                        if isinstance(element, dict) and "text" in element:
                            texte_description = element["text"]
                            break
                        # Si l'élément est juste une string dans la liste
                        elif isinstance(element, str):
                            texte_description += element

                # Création du Document LangChain avec une vraie String
                doc = Document(
                    page_content=texte_description,
                    metadata={"source": str(img_path), "type": "image", "format": img_path.suffix}
                )
                docs_images.append(doc)

                # ⏱️ Pause de 4 secondes pour éviter le ban 429 RESOURCE_EXHAUSTED
                time.sleep(4)

            except Exception as e:
                if "429" in str(e):
                    print(f"\n⚠️ Quota API atteint ! Pause forcée de 60 secondes...")
                    time.sleep(60)
                else:
                    print(f"\n❌ Erreur sur {img_path.name}: {e}")

        print(f"\n✅ Terminé : {len(docs_images)} images analysées avec succès !")

        # Affiche la première description pour vérifier que tout marche bien
        if docs_images:
            print("\nExemple de la première description extraite :")
            print("-" * 50)
            print(docs_images[0].page_content)


if __name__ == '__main__':
    ch = Chunker(data_base=Path('/home/gematura/cursus_42/rag/data/vllm-0.10.1/'))
    # print(ch.list_img_suffix)
    ch.loader_database()
