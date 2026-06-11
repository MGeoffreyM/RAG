import magic
from pathlib import Path

dossier = Path('/home/gematura/cursus_42/rag/data/vllm-0.10.1')
list_text = []
list_img = []
list_other = []


def detecter_type_fichier(chemin_fichier):
    # Crée un analyseur de type MIME
    mime = magic.Magic(mime=True)
    type_mime = mime.from_file(chemin_fichier)
    
    print(f"Type MIME détecté : {type_mime}")
    
    if type_mime.startswith('text/'):
        return "C'est un fichier texte."
    elif type_mime.startswith('image/'):
        return "C'est une image."
    else:
        return "C'est un autre type de fichier (binaire, audio, pdf, etc.)."
for element in dossier.rglob('*'):
    s = element.suffix
    if s not in list_suffix:
        list_suffix.append(s)
print(list_suffix)