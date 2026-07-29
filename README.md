*This project has been created as part of the 42 curriculum by gematura.*

**🇫🇷 [Version Française](README_fr.md)**

# RAG Against the Machine

## 1. Description
"RAG Against the Machine" is a comprehensive Retrieval-Augmented Generation (RAG) system. Its goal is to ingest, index, and query a large codebase (the vLLM repository) to answer technical questions accurately. Instead of relying solely on the pre-trained knowledge of a Large Language Model (LLM), this system retrieves relevant code snippets and documentation directly from the source and feeds them to a local model (Qwen/Qwen3-0.6B) to generate highly grounded, factual answers.

## 2. System Architecture
The pipeline is designed to be modular and robust, built around four main stages:
1. **Ingestion & Chunking (`chunking.py`)**: Reads the raw corpus, detects file types (code, markdown, images), and splits them into meaningful segments.
2. **Indexing (`retriever.py`)**: Stores the chunks into a BM25 lexical index and a ChromaDB vector database for semantic search.
3. **Retrieval (`retriever.py`)**: Receives a user query, processes it through both BM25 and ChromaDB, and fuses the rankings using Reciprocal Rank Fusion (RRF) to retrieve the top `k` most relevant sources.
4. **Generation (`llm.py` & `retriever.py`)**: Passes the retrieved context and the user query to the local LLM to generate a final answer. Data validation across all stages is strictly enforced using Pydantic models (`tools.py`).

## 3. Chunking Strategy
Document segmentation is crucial for accurate retrieval. The system uses Langchain's `RecursiveCharacterTextSplitter` with adaptive strategies based on file extensions:
* **Language-Specific Splitting**: Custom separators are used for Python (`.py`), C++ (`.cpp`, `.h`), JavaScript (`.js`), and Markdown (`.md`) to preserve syntactic boundaries (e.g., keeping functions and classes intact).
* **Constraints**: The maximum chunk size is strictly limited to 2000 characters (configurable via `--max_chunk_size`), with a 25% overlap to maintain context between adjacent chunks.
* **Image Processing (Bonus)**: For image files, a local vision model (`SmolVLM-256M-Instruct`) is used to generate text descriptions, which are then chunked and indexed alongside textual data.
## 4. Retrieval Method
The system employs a **Hybrid Retrieval** approach, combining two distinct algorithms to maximize search accuracy:

* **Lexical Retrieval (BM25)**: Uses the `bm25s` library with English stemming and stop-word removal. It is highly effective for exact keyword matches. BM25 evaluates the relevance of a document $D$ to a query $Q$ (containing keywords $q_i$) using term frequency and document length normalization:
  $$ \text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)} $$
  *(Where $f(q_i, D)$ is the term frequency, $|D|$ is the document length, and the tuning parameters are set to $k_1=1.5$ and $b=0.75$).*

* **Semantic Retrieval (ChromaDB + Embeddings)**: Uses the `all-MiniLM-L6-v2` CPU-friendly embedding model to understand the *meaning* of the query. It maps texts into dense high-dimensional vectors and retrieves documents based on their spatial proximity, easily capturing paraphrased questions where exact keywords fail.

* **Ranking Mechanism (Weighted RRF)**: The results from both methods are unified using a custom **Reciprocal Rank Fusion (RRF)**. Instead of relying on raw scores which have different scales, RRF relies solely on the rank $r(d)$ of the document in each search system. We apply a weighted formula:
  $$ \text{Score}_{\text{Hybrid}}(d) = w_{\text{BM25}} \cdot \frac{1}{K + r_{\text{BM25}}(d)} + w_{\text{Chroma}} \cdot \frac{1}{K + r_{\text{Chroma}}(d)} $$
  *(Where $K=10$ acts as a smoothing constant to mitigate the extreme weight of top-ranked items, with a lexical weight $w_{\text{BM25}} = 0.60$ and a semantic weight $w_{\text{Chroma}} = 0.40$).*
## 5. Performance Analysis
The system is optimized to run entirely on CPU-only machines:
* **Indexing Time**: The incremental indexing architecture ensures that a full corpus ingestion is completed well within the 5-minute limit, while subsequent updates are near-instantaneous.
* **Retrieval Throughput**: Heavy operations are optimized. The system comfortably retrieves 200 questions in under 90 seconds.
* **Recall@5**: Thanks to the hybrid retrieval, the system exceeds the baseline requirements, comfortably achieving `> 80%` on documentation questions and `> 50%` on code questions. Smaller chunks with high overlap significantly improved recall for code datasets. *(Note: specific local metrics may vary slightly based on hardware).*

## 6. Design Decisions
* **Strict Pydantic Validation**: All data exchanged between components uses Pydantic models to ensure absolute type safety, preventing silent failures during retrieval and generation.
* **Graceful Degradation for Edge Cases**: The CLI leverages Pydantic's `@validate_call` to catch empty queries or invalid `k` parameters (`k=0`). These are handled safely without unhandled Python tracebacks, ensuring extreme system reliability.
* **CPU Optimization**: Both the embedding model and the generation LLM are configured to load efficiently on CPU if CUDA is unavailable, conforming to campus machine constraints.

## 7. Challenges Faced
* **Context Length Limits**: Fitting retrieved sources within the prompt without exceeding the LLM's context window. This was solved by strictly enforcing the 2000-character chunk limit and carefully structuring the prompt template.
* **Semantic Search Speed**: Embedding the entire corpus on a CPU was initially slow. This was resolved by implementing an incremental indexing feature and processing documents in batches with `tqdm` progress bars.

## 8. Bonus Features Implemented
This project successfully integrates all 5 mandatory bonuses:
1. **Semantic Embeddings**: Implemented via `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in ChromaDB.
2. **Hybrid Retrieval**: Combines BM25 and Semantic search using the RRF algorithm.
3. **Incremental Indexing**: Uses file modification times (`mtime`) and state tracking (`index_state.json`) to re-index only modified, added, or deleted files.
4. **Caching**: Queries and answers are cached in a JSON file using deterministic `UUID5` hashing based on the query and `k`, drastically reducing repeated query times.
5. **Local HTTP API**: A `FastAPI` server is available to interact with the RAG pipeline via REST endpoints, complete with Swagger documentation.

## 9. Instructions
### Installation
Ensure you have `uv` and Python 3.10+ installed.
```bash
make install
```
*(This command runs `uv sync` to install all dependencies from `uv.lock`)*

### Makefile Rules
The project includes a comprehensive `Makefile` to automate common tasks:

* `make install`: Installs all project dependencies using `uv sync`.
* `make run`: Executes the full RAG pipeline. It indexes the data, runs the search and answer generation for the datasets (code and docs), and evaluates the results.
* `make retrieval`: Executes only the retrieval part of the pipeline (indexing, searching the datasets, and performance evaluation), skipping the LLM answer generation.
* `make debug`: Runs the program in interactive debug mode using `pdb`.
* `make lint`: Analyzes code quality using `flake8` and checks static typing with `mypy` (using the subject's required warning flags).
* `make lint-strict`: Runs `flake8` and `mypy` in strict mode (`--strict`) for rigorous type checking.
* `make clean`: Cleans the workspace by removing cache artifacts (`__pycache__`, `.mypy_cache`) and the generated data directories (`data/processed` and `data/output`).
* `make fclean`: Executes the `clean` rule and additionally removes the local virtual environments (`.venv` and those stored in the `goinfre` folder).

## 10. Example Usage
The CLI is built with Python Fire. All paths and parameters are fully configurable.

**Index the corpus:**
```bash
uv run python -m src index --data_path data/raw/vllm-0.10.1/ --max_chunk_size 2000
```
*(Use `--force` to force a full re-index, or `--image` to enable Vision LLM processing)*

**Search for sources (single query):**
```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

**Generate an answer (single query):**
```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

**Batch processing:**
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions/

uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions/
```

**Launch the Local API:**
```bash
uv run python -m src serve --port 8000
```
*Access the API documentation at `http://127.0.0.1:8000/docs`.*

## 11. Resources & AI Usage
**Resources:**
* [BM25 algorithm documentation](https://en.wikipedia.org/wiki/Okapi_BM25)
* [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
* [Reciprocal Rank Fusion (RRF) Concepts](https://plg.uwaterloo.ca/~gvcormac/cormackcrv09.pdf)
* [Hugging Face Models & Datasets](https://huggingface.co/)

**AI Usage Statement:**
AI tools (Large Language Models) were utilized during this project strictly adhering to the school's guidelines:
* **Task Automation**: Writing README files and translating project documents.
* **Brainstorming**: Discussing the theoretical differences between BM25 and Semantic search to write an efficient Hybrid retrieval function (RRF).
* **Code Review**: Utilizing AI as a secondary linting tool to suggest improvements for `flake8` and `mypy` compliance before peer evaluation. 
No core logic was blindly copy-pasted; all AI-assisted features were deeply tested, refactored, and are fully understood by the author.
