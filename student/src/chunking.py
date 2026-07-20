import os
import base64
import time
from dotenv import load_dotenv
from pathlib import Path
import magic
from tqdm import tqdm
from pydantic import BaseModel, model_validator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from src.tools import MinimalSource


load_dotenv()


class Chunker(BaseModel):
    data_base: Path
    list_txt: list[str] = []
    list_img: list[Path] = []
    list_doc: list[Document] = []
    list_chunk: list[MinimalSource] = []

    @model_validator(mode='after')
    def find_suffix(self) -> 'Chunker':
        txt = ['.json', '.yaml', '.yml', '.toml', '.xml', '.sh', '.bash',
               '.md', '.txt']
        bin_extensions = ['.pyc', '.so', '.pyd', '.db', '.sqlite', '.a']

        for element in tqdm(self.data_base.rglob('*'),
                            desc="Extraction des fichiers"):
            if element.is_file():
                try:
                    type_mime = magic.from_file(str(element), mime=True)
                except Exception:
                    continue
                s = element.suffix.lower()

                if type_mime.startswith('text/'):
                    if s and s not in self.list_txt \
                         and s not in bin_extensions:
                        self.list_txt.append(s)
                elif type_mime.startswith('application/'):
                    if s not in self.list_txt and s in txt:
                        self.list_txt.append(s)
                elif type_mime.startswith('image/'):
                    if s != ".ico":
                        self.list_img.append(element)
        return self

    def process_text(self) -> None:
        if not self.list_txt:
            print("Aucun fichier texte détecté.")
            return

        loader = GenericLoader.from_filesystem(
            self.data_base,
            glob="**/*",
            suffixes=self.list_txt,
            parser=TextParser()
        )
        try:
            docs_text = loader.load()
            self.list_doc.extend(docs_text)

        except Exception as e:
            print(f"Erreur lors du parsing des fichiers : {e}")

    def _encode_image(self, image_path: Path) -> str:
        """Encode une image en Base64 pour l'envoyer au modèle via l'API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_images(self) -> None:
        llm_vision = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
        docs_images = []

        for img_path in tqdm(self.list_img, desc="Analyse des images"):
            try:
                file_size_kb = os.path.getsize(img_path) / 1024
                if file_size_kb > 4000:
                    print("\nFichier ignoré (Trop lourd pour Groq, > 4 Mo)"
                          f" : {img_path.name}")
                    continue

                base64_image = self._encode_image(img_path)

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"
                                          f"{base64_image}"}
                        }
                    ]
                )

                reponse = llm_vision.invoke([message])

                texte_description = ""
                if isinstance(reponse.content, str):
                    texte_description = reponse.content
                elif isinstance(reponse.content, list):
                    for element in reponse.content:
                        if isinstance(element, dict) and "text" in element:
                            texte_description = element["text"]
                            break
                        elif isinstance(element, str):
                            texte_description += element

                doc = Document(
                    page_content=texte_description,
                    metadata={"source": str(img_path), "type": "image",
                              "format": img_path.suffix}
                )
                docs_images.append(doc)
                time.sleep(2)

            except Exception as e:
                error_msg = str(e)
                print(error_msg)
                if "429" in error_msg:
                    print("\nQuota Groq atteint ! Pause de 60 secondes...")
                    time.sleep(60)
                else:
                    print(f"\nErreur sur {img_path.name}: {e}")

        self.list_doc.extend(docs_images)

    def create_smart_chunks(self, chunk_size: int) -> None:
        chunk_overlap = int(chunk_size * 0.25)

        splitters = {
            "default": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            ),
            ".py": RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            ),
            ".cpp": RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            ),
            ".h": RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            ),
            ".md": RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            ),
            ".js": RecursiveCharacterTextSplitter.from_language(
                language=Language.JS, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True, strip_whitespace=False
            )
        }

        for doc in tqdm(self.list_doc, desc="Decoupage en chunks"):
            source_path = doc.metadata.get("source", "")
            ext = doc.metadata.get("format", "")

            if not ext and "." in source_path:
                ext = "." + source_path.split(".")[-1].lower()

            splitter = splitters.get(ext, splitters["default"])

            chunks = splitter.split_documents([doc])
            for elem in chunks:
                raw_path = str(elem.metadata['source']).replace("\\", "/")
                self.list_chunk.append(MinimalSource(
                    file_path=raw_path,
                    first_character_index=elem.metadata['start_index'],
                    last_character_index=len(elem.page_content)
                    + elem.metadata['start_index'],
                    chunk_content=elem.page_content))
