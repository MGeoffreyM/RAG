BLUE = \033[1;34m
GREEN = \033[1;32m
RESET = \033[0m

.PHONY: install run debug clean lint lint-strict

install:
	@printf '$(BLUE)Installation des dépendances avec uv...$(RESET)\n'
	uv sync

run:
	@printf '$(BLUE)Lancement de RAG$(RESET)\n'
	uv run python -m src

debug:
	@printf '$(BLUE)Lancement en mode debug de RAG...$(RESET)\n'
	uv run python -m pdb src

clean:
	@printf '$(BLUE)Nettoyage des fichiers temporaires...$(RESET)\n'
	rm -rf .mypy_cache
	rm -r /home/gematura/cursus_42/rag/data/processed
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv
	rm -rf /home/gematura/goinfre/mes_venvs/rag_venv

lint:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	uv run flake8 .
	@printf '$(BLUE)Lancement de mypy (règles du sujet)...$(RESET)\n'
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .
	@printf '$(GREEN)MYPY et Flake8 terminés. Aucune erreur trouvée.$(RESET)\n'

lint-strict:
	@printf '$(BLUE)Lancement de flake8...$(RESET)\n'
	uv run flake8 .
	@printf '$(BLUE)Lancement de mypy (mode strict)...$(RESET)\n'
	uv run mypy --strict .
	@printf '$(GREEN)MYPY et Flake8 terminés en mode STRICT. Aucune erreur trouvée.$(RESET)\n'