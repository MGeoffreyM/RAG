from pathlib import Path
import magic
from tqdm import tqdm
from pydantic import BaseModel, model_validator
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from src.tools import MinimalSource
from src.llm import VisionLlm


class Chunker(BaseModel):
    data_base: Path
    new_files: list[Path] | None = None
    list_txt: list[str] = []
    list_img: list[Path] = []
    list_doc: list[Document] = []
    list_chunk: list[MinimalSource] = []

    @model_validator(mode='after')
    def find_suffix(self) -> 'Chunker':
        """
        Valide et trie les fichiers trouvés après l'initialisation du modèle.

        Cette méthode parcourt les fichiers définis dans 'new_files' ou,
        par défaut, tous les fichiers du répertoire 'data_base'.
        Elle utilise 'magic' pour déterminer le type MIME de chaque fichier.
        Elle ignore les extensions binaires spécifiques et peuple 'list_txt'
        avec les extensions de fichiers textuels et 'list_img' avec les
        chemins des images valides.

        Returns:
            Chunker: L'instance de la classe mise à jour avec les listes
            d'extensions et d'images remplies.
        """
        txt = ['.json', '.yaml', '.yml', '.toml', '.xml', '.sh', '.bash',
               '.md', '.txt']
        bin_extensions = ['.pyc', '.so', '.pyd', '.db', '.sqlite', '.a']

        files = self.new_files if self.new_files is not None else list(
            self.data_base.rglob('*'))
        for element in tqdm(files, desc="Extraction des fichiers"):
            if element.is_file():
                try:
                    type_mime = magic.from_file(str(element), mime=True)
                except Exception:
                    print("élément ignoré")
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
        """
        Charge et analyse les fichiers textuels identifiés.

        Si des fichiers sont spécifiés dans 'new_files', la méthode utilise
        'TextLoader' pour charger individuellement ceux qui ont une extension
        valide.
        Sinon, elle utilise 'GenericLoader' pour charger de manière globale
        tous les fichiers correspondants depuis le répertoire 'data_base'.
        Les documents extraits sont ensuite ajoutés à l'attribut 'list_doc'.

        Returns:
            None
        """
        if not self.list_txt:
            print("Aucun fichier texte détecté.")
            return

        if self.new_files is not None:
            for file in self.new_files:
                if file.suffix.lower() in self.list_txt:
                    try:
                        doc = TextLoader(str(file)).load()
                        self.list_doc.extend(doc)
                    except Exception as e:
                        print(f"Erreur au chargement de {file}: {e}")
        else:
            try:
                loader = GenericLoader.from_filesystem(
                    self.data_base,
                    glob="**/*",
                    suffixes=self.list_txt,
                    parser=TextParser()
                )
                docs_text = loader.load()
                self.list_doc.extend(docs_text)

            except Exception as e:
                print(f"Erreur lors du parsing des fichiers : {e}")

    def process_images(self) -> None:
        """
        Analyse les images identifiées et génère des descriptions textuelles.

        Cette méthode itère sur les chemins d'images stockés dans 'list_img'.
        Elle utilise 'VisionLlm' pour obtenir une description textuelle de
        chaque image.
        Elle crée ensuite un objet 'Document' contenant cette description et
        des métadonnées (source, type, format), qu'elle ajoute à 'list_doc'.

        Returns:
            None
        """
        llm_vision = VisionLlm()
        docs_images = []

        for img_path in tqdm(self.list_img, desc="Analyse des images"):
            reponse = llm_vision.describe_image(str(img_path))

            texte_description = reponse

            doc = Document(
                page_content=texte_description,
                metadata={"source": str(img_path), "type": "image",
                          "format": img_path.suffix}
            )
            docs_images.append(doc)

        self.list_doc.extend(docs_images)

    def create_smart_chunks(self, chunk_size: int) -> None:
        """
        Découpe les documents chargés en segments (chunks) adaptés à leur
        format.

        La méthode calcule d'abord un chevauchement ('chunk_overlap')
        correspondant à 25% de la taille du chunk.
        Elle sélectionne ensuite le 'RecursiveCharacterTextSplitter' approprié
        en fonction de l'extension du fichier, avec des règles spécifiques
        pour Python, C++, Markdown et JavaScript.
        Les segments générés sont convertis en objets 'MinimalSource'
        (contenant le chemin, l'index de départ, l'index de fin et le contenu)
        et ajoutés à 'list_chunk'.

        Args:
            chunk_size (int): La taille maximale en caractères d'un chunk.

        Returns:
            None
        """
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
