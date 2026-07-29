*Ce projet a été réalisé dans le cadre du cursus 42 par gematura.*

**🌐 [English Version](README.md)**

# RAG Against the Machine

## 1. Description
"RAG Against the Machine" est un système complet de Génération Augmentée par la Recherche (RAG). Son objectif est d'ingérer, d'indexer et d'interroger une large base de code (le dépôt vLLM) pour répondre précisément à des questions techniques. Au lieu de s'appuyer uniquement sur les connaissances pré-entraînées d'un Grand Modèle de Langage (LLM), ce système récupère des extraits de code et de la documentation pertinents directement depuis la source et les fournit à un modèle local (Qwen/Qwen3-0.6B) pour générer des réponses factuelles et bien fondées.

## 2. Architecture du système
Le pipeline est conçu pour être modulaire et robuste, articulé autour de quatre étapes principales :
1. **Ingestion & Découpage (`chunking.py`)** : Lit le corpus brut, détecte les types de fichiers (code, markdown, images) et les découpe en segments pertinents.
2. **Indexation (`retriever.py`)** : Stocke les morceaux (chunks) dans un index lexical BM25 et une base de données vectorielle ChromaDB pour la recherche sémantique.
3. **Recherche (`retriever.py`)** : Reçoit une requête utilisateur, la traite via BM25 et ChromaDB, et fusionne les classements à l'aide de la fusion de rangs réciproque (RRF) pour récupérer les `k` sources les plus pertinentes.
4. **Génération (`llm.py` & `retriever.py`)** : Transmet le contexte récupéré et la requête de l'utilisateur au LLM local pour générer une réponse finale. La validation des données à toutes les étapes est strictement appliquée à l'aide des modèles Pydantic (`tools.py`).

## 3. Stratégie de découpage (Chunking)
La segmentation des documents est cruciale pour une recherche précise. Le système utilise le `RecursiveCharacterTextSplitter` de Langchain avec des stratégies adaptatives basées sur les extensions de fichiers :
* **Découpage spécifique au langage** : Des séparateurs personnalisés sont utilisés pour Python (`.py`), C++ (`.cpp`, `.h`), JavaScript (`.js`) et Markdown (`.md`) afin de préserver les limites syntaxiques (ex: garder les fonctions et classes intactes).

* **Contraintes** : La taille maximale d'un chunk est strictement limitée à 2000 caractères (configurable via `--max_chunk_size`), avec un chevauchement (overlap) de 25% pour maintenir le contexte entre les chunks adjacents.

* **Traitement des images (Bonus)** : Pour les fichiers images, un modèle de vision local (`SmolVLM-256M-Instruct`) est utilisé pour générer des descriptions textuelles, qui sont ensuite découpées et indexées avec les données textuelles.

## 4. Méthode de recherche
Le système utilise une approche de **Recherche Hybride**, combinant deux algorithmes distincts pour maximiser la précision :

* **Recherche Lexicale (BM25)** : Utilise la bibliothèque `bm25s` avec la racinisation (stemming) anglaise et la suppression des mots vides. Elle est très efficace pour les correspondances exactes de mots-clés. BM25 estime la pertinence d'un document $D$ pour une requête $Q$ (contenant les mots-clés $q_i$) via la fréquence des termes et la normalisation de la longueur du document :
  $$ \text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)} $$
  *(Où $f(q_i, D)$ est la fréquence du terme, $|D|$ est la longueur du document, avec des paramètres d'optimisation fixés à $k_1=1.5$ et $b=0.75$).*

* **Recherche Sémantique (ChromaDB + Embeddings)** : Utilise le modèle `all-MiniLM-L6-v2` (optimisé CPU) pour comprendre le *sens* de la requête. Il projette les textes sous forme de vecteurs denses et récupère les documents par proximité spatiale, ce qui permet de capturer les questions paraphrasées.

* **Mécanisme de classement (RRF Pondéré)** : Les résultats des deux méthodes sont unifiés en utilisant une variante de la **Fusion de Rangs Réciproque (RRF)**. Au lieu de s'appuyer sur les scores bruts qui n'ont pas la même échelle, le RRF utilise le rang $r(d)$ de chaque document. Une formule pondérée est appliquée :
  $$ \text{Score}_{\text{Hybride}}(d) = w_{\text{BM25}} \cdot \frac{1}{K + r_{\text{BM25}}(d)} + w_{\text{Chroma}} \cdot \frac{1}{K + r_{\text{Chroma}}(d)} $$
  *(Où $K=10$ agit comme une constante de lissage pour atténuer le poids des tout premiers résultats, avec un poids lexical $w_{\text{BM25}} = 0.60$ et un poids sémantique $w_{\text{Chroma}} = 0.40$).*
## 5. Analyse des performances
Le système est optimisé pour fonctionner entièrement sur des machines limitées au CPU :
* **Temps d'indexation** : L'architecture d'indexation incrémentale garantit qu'une ingestion complète du corpus se termine bien en deçà de la limite de 5 minutes, tandis que les mises à jour ultérieures sont quasi instantanées.
* **Débit de recherche** : Les opérations lourdes sont optimisées. Le système récupère confortablement 200 questions en moins de 90 secondes.
* **Recall@5** : Grâce à la recherche hybride, le système dépasse les exigences de base, atteignant confortablement `> 80%` sur les questions de documentation et `> 50%` sur les questions de code. Des chunks plus petits avec un chevauchement élevé ont considérablement amélioré le rappel pour les jeux de données de code. *(Note : les métriques locales spécifiques peuvent varier légèrement en fonction du matériel).*

## 6. Décisions de conception
* **Validation stricte avec Pydantic** : Toutes les données échangées entre les composants utilisent des modèles Pydantic pour garantir une sécurité absolue des types, évitant les échecs silencieux pendant la recherche et la génération.
* **Dégradation gracieuse pour les cas limites** : Le CLI exploite `@validate_call` de Pydantic pour intercepter les requêtes vides ou les paramètres `k` invalides (`k=0`). Ceux-ci sont gérés en toute sécurité sans tracebacks Python non gérés, assurant une fiabilité extrême du système.
* **Optimisation CPU** : Le modèle d'intégration (embedding) et le LLM de génération sont tous configurés pour se charger efficacement sur le CPU si CUDA n'est pas disponible, conformément aux contraintes des machines du campus.

## 7. Défis rencontrés
* **Limites de longueur du contexte** : Faire tenir les sources récupérées dans le prompt sans dépasser la fenêtre de contexte du LLM. Cela a été résolu en appliquant strictement la limite de 2000 caractères par chunk et en structurant soigneusement le modèle de prompt.
* **Vitesse de recherche sémantique** : L'intégration de l'ensemble du corpus sur un CPU était initialement lente. Cela a été résolu en implémentant une fonctionnalité d'indexation incrémentale et en traitant les documents par lots avec des barres de progression `tqdm`.

## 8. Fonctionnalités Bonus Implémentées
Ce projet intègre avec succès les 5 bonus :
1. **Embeddings Sémantiques** : Implémenté via `sentence-transformers` (`all-MiniLM-L6-v2`) et stocké dans ChromaDB.
2. **Recherche Hybride** : Combine BM25 et la recherche sémantique en utilisant l'algorithme RRF.
3. **Indexation Incrémentale** : Utilise les temps de modification des fichiers (`mtime`) et le suivi de l'état (`index_state.json`) pour ne réindexer que les fichiers modifiés, ajoutés ou supprimés.
4. **Mise en cache** : Les requêtes et les réponses sont mises en cache dans un fichier JSON à l'aide d'un hachage déterministe `UUID5` basé sur la requête et `k`, réduisant considérablement les temps de requêtes répétées.
5. **API HTTP Locale** : Un serveur `FastAPI` est disponible pour interagir avec le pipeline RAG via des points de terminaison REST, complet avec la documentation Swagger.

## 9. Instructions
### Installation
Assurez-vous d'avoir installé `uv` et Python 3.10+.
```bash
make install
```
*(Cette commande exécute `uv sync` pour installer toutes les dépendances à partir de `uv.lock`)*

### Règles du Makefile
Le projet inclut un `Makefile` complet pour automatiser les tâches courantes :

* `make install` : Installe toutes les dépendances du projet de manière isolée en utilisant `uv sync`.
* `make run` : Exécute le pipeline RAG complet. Cette commande indexe les données, lance la recherche et génère les réponses pour les jeux de données (code et documentation), puis évalue automatiquement vos résultats.
* `make retrieval` : Exécute uniquement la partie "Recherche" (indexation, exécution des requêtes sur les datasets, et évaluation des performances), sans lancer la génération de réponses par le LLM.
* `make debug` : Lance le programme en mode débogage interactif avec le module `pdb`.
* `make lint` : Analyse la qualité du code avec `flake8` et vérifie le typage statique avec `mypy` (en utilisant les drapeaux d'avertissements requis par le sujet).
* `make lint-strict` : Exécute `flake8` et `mypy` en mode strict (`--strict`) pour une vérification approfondie.
* `make clean` : Nettoie l'espace de travail en supprimant les caches (`__pycache__`, `.mypy_cache`), ainsi que les répertoires de données générés (`data/processed` et `data/output`).
* `make fclean` : Exécute la règle `clean` et supprime en plus les environnements virtuels locaux (dossier `.venv` et ceux stockés dans le dossier `goinfre`).

## 10. Exemple d'utilisation
Le CLI est construit avec Python Fire. Tous les chemins et paramètres sont entièrement configurables.

**Indexer le corpus :**
```bash
uv run python -m src index --data_path data/raw/vllm-0.10.1/ --max_chunk_size 2000
```
*(Utilisez `--force` pour forcer une réindexation complète, ou `--image` pour activer le traitement par Vision LLM)*

**Rechercher des sources (requête unique) :**
```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

**Générer une réponse (requête unique) :**
```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

**Traitement par lots :**
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions/

uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions/
```

**Lancer l'API Locale :**
```bash
uv run python -m src serve --port 8000
```
*Accédez à la documentation de l'API sur `http://127.0.0.1:8000/docs`.*

## 11. Ressources & Utilisation de l'IA
**Ressources :**
* [Documentation de l'algorithme BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
* [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
* [Concepts de Fusion de Rangs Réciproque (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormackcrv09.pdf)
* [Modèles et jeux de données Hugging Face](https://huggingface.co/)

**Déclaration d'utilisation de l'IA :**
Des outils d'IA (Grands Modèles de Langage) ont été utilisés au cours de ce projet en respectant strictement les directives de l'école :
* **Automatisation des tâches** : Rédaction des fichiers README et traduction de documents.
* **Réflexion (Brainstorming)** : Discussion sur les différences théoriques entre BM25 et la recherche sémantique pour écrire une fonction de recherche hybride efficace (RRF).
* **Revue de code** : Utilisation de l'IA comme outil de linting secondaire pour suggérer des améliorations pour la conformité `flake8` et `mypy` avant l'évaluation par les pairs.
Aucune logique principale n'a été copiée-collée aveuglément ; toutes les fonctionnalités assistées par l'IA ont été testées en profondeur, refactorisées et sont parfaitement comprises par l'auteur.
