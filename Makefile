BLUE = \033[1;34m
GREEN = \033[1;32m
RESET = \033[0m
MAP ?= maps/easy/01_linear_path.txt

.PHONY: install run debug clean lint lint-strict

install:
	@printf '$(BLUE)Installation des dépendances avec uv...$(RESET)\n'
	uv sync

run:
	@printf '$(BLUE)Lancement du simulateur sur la map : $(MAP)...$(RESET)\n'
	uv run python main.py $(MAP)

visual:
	@printf '$(BLUE)Lancement du simulateur en mode visualisation sur la map : $(MAP)...$(RESET)\n'
	uv run python main.py --gui $(MAP)

debug:
	@printf '$(BLUE)Lancement en mode debug sur la map : $(MAP)...$(RESET)\n'
	uv run python -m pdb main.py $(MAP)

clean:
	@printf '$(BLUE)Nettoyage des fichiers temporaires...$(RESET)\n'
	rm -rf .mypy_cache
	find . -type d -name '__pycache__' -exec rm -rf {} +
	uv clean

fclean: clean
	@printf "$(BLUE)Suppression de l'environnement virtuel...$(RESET)\n"
	rm -rf .venv

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