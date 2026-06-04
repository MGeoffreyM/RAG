from pathlib import Path


dossier = Path('/home/gematura/cursus_42/tronc_commum/rag/data/vllm-0.10.1')

if __name__ == '__main__':
    for fichier in dossier.rglob('*.py'):
        print(fichier)