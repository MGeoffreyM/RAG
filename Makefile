BLUE = \033[1;34m
GREEN = \033[1;32m
RESET = \033[0m

.PHONY: install run retrieval debug clean lint lint-strict

install:
	@printf '$(BLUE)Installation des dépendances avec uv...$(RESET)\n'

	uv sync

run:
	@printf '$(BLUE)Lancement de RAG$(RESET)\n'
	@printf '$(GREEN)Indexation des données$(RESET)\n'
	uv run python -m src index
	@printf '$(GREEN)Traitement des questions de code$(RESET)\n'
	uv run python -m src search_dataset --dataset_path="../data/datasets/datasets_public/public/UnansweredQuestions/dataset_code_public.json"
	uv run python -m src answer_dataset --student_search_results_path="data/output/search_results/dataset_code_public.json"
	@printf '$(GREEN)Traitement des questions de texte$(RESET)\n'
	uv run python -m src search_dataset --dataset_path="../data/datasets/datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
	uv run python -m src answer_dataset --student_search_results_path="data/output/search_results/dataset_docs_public.json"
	@printf '$(GREEN)Evaluation code$(RESET)\n'
	uv run python -m src evaluate --student_search_results_path="data/output/search_results/dataset_code_public.json" --dataset_path="../data/datasets/datasets_public/public/AnsweredQuestions/dataset_code_public.json"
	@printf '$(GREEN)Evaluation docs$(RESET)\n'
	uv run python -m src evaluate --student_search_results_path="data/output/search_results/dataset_docs_public.json" --dataset_path="../data/datasets/datasets_public/public/AnsweredQuestions/dataset_docs_public.json"

retrieval:
	@printf '$(BLUE)Lancement de RAG$(RESET)\n'
	@printf '$(GREEN)Indexation des données$(RESET)\n'
	uv run python -m src index
	uv run python -m src search_dataset --dataset_path="../data/datasets/datasets_public/public/UnansweredQuestions/dataset_code_public.json"
	uv run python -m src search_dataset --dataset_path="../data/datasets/datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
	@printf '$(GREEN)Evaluation code$(RESET)\n'
	uv run python -m src evaluate --student_search_results_path="data/output/search_results/dataset_code_public.json" --dataset_path="../data/datasets/datasets_public/public/AnsweredQuestions/dataset_code_public.json"
	@printf '$(GREEN)Evaluation docs$(RESET)\n'
	uv run python -m src evaluate --student_search_results_path="data/output/search_results/dataset_docs_public.json" --dataset_path="../data/datasets/datasets_public/public/AnsweredQuestions/dataset_docs_public.json"

debug:
	@printf '$(BLUE)Lancement en mode debug de RAG...$(RESET)\n'
	uv run python -m pdb src

clean:
	@printf '$(BLUE)Nettoyage des fichiers...$(RESET)\n'
	rm -rf .mypy_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean
	- rm -r ../evaluations
	- rm -r data/processed
	- rm -r data/output

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv
	rm -rf /home/$(USER)/goinfre/mes_venvs/rag_venv

lint:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	- uv run flake8 .
	@printf '$(BLUE)Lancement de mypy...$(RESET)\n'
	- uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .
	@printf '$(GREEN)MYPY et Flake8 terminés.$(RESET)\n'

lint-strict:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	- uv run flake8 .
	@printf '$(BLUE)Lancement de mypy (mode strict)...$(RESET)\n'
	- uv run mypy --strict .
	@printf '$(GREEN)MYPY et Flake8 terminés en mode STRICT. Aucune erreur trouvée.$(RESET)\n'