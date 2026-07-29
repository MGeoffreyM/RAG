import json
from pathlib import Path
from pydantic import BaseModel
from typing import Any


class Evaluator(BaseModel):
    """
    Permet d'évaluer les performances de récupération d'un système RAG.
    """
    student_search_results_path: Path
    dataset_path: Path

    def _calculate_overlap(self, retrieved: dict[str, Any],
                           expected: dict[str, Any]) -> Any:
        """
        Calcule la proportion de chevauchement entre un segment de texte
        récupéré et le segment attendu.

        Cette méthode vérifie d'abord si les chemins des fichiers
        correspondent.
        Elle calcule ensuite l'intersection entre les indices de début
        et de fin des deux sources.
        Le résultat retourné correspond au ratio de la longueur du
        chevauchement par rapport à la longueur totale de la source attendue.

        Args:
            retrieved (dict[str, Any]): Le dictionnaire contenant les
                métadonnées de la source récupérée, incluant le chemin et les
                indices de caractères.
            expected (dict[str, Any]): Le dictionnaire contenant les
                métadonnées de la source de vérité terrain (Ground Truth).

        Returns:
            Any (float): Le pourcentage de chevauchement,
                généralement un float compris entre 0.0 et 1.0.
        """
        ret_path = Path(retrieved.get("file_path", "")).as_posix()
        exp_path = Path(expected.get("file_path", "")).as_posix()

        if not ret_path.endswith(exp_path) and not exp_path.endswith(ret_path):
            return 0.0

        r_start = retrieved.get("first_character_index", 0)
        r_end = retrieved.get("last_character_index", 0)
        e_start = expected.get("first_character_index", 0)
        e_end = expected.get("last_character_index", 0)

        overlap_start = max(r_start, e_start)
        overlap_end = min(r_end, e_end)
        overlap_length = max(0, overlap_end - overlap_start)

        expected_length = e_end - e_start
        if expected_length == 0:
            return 0.0

        return overlap_length / expected_length

    def _evaluate_question_at_k(self, student_results: list[dict[str, Any]],
                                expected_sources: list[dict[str, Any]],
                                k: int) -> float:
        """
        Évalue le score (recall) pour une question spécifique en se limitant
        aux `k` premiers résultats récupérés.

        La méthode compare les sources récupérées, limitées au nombre `k`,
        avec les sources attendues.
        Une source est considérée comme trouvée si le chevauchement calculé
        avec la source attendue est supérieur ou égal à 5% (0.05).
        Elle retourne la proportion de sources trouvées par rapport au nombre
        total de sources attendues.

        Args:
            student_results (list[dict[str, Any]]): La liste des sources
                extraites par le système évalué.
            expected_sources (list[dict[str, Any]]): La liste des sources
                exactes attendues pour la question donnée.
            k (int): Le nombre maximum de résultats de recherche à analyser.

        Returns:
            float: Le score de rappel pour cette question,
            où 0.0 indique qu'aucune source n'est trouvée.
        """
        if not expected_sources:
            return 0.0

        top_k_retrieved = student_results[:k]
        found_count = 0

        for expected in expected_sources:
            is_found = False
            for retrieved in top_k_retrieved:
                overlap = self._calculate_overlap(retrieved, expected)
                if overlap >= 0.05:
                    is_found = True
                    break

            if is_found:
                found_count += 1

        return found_count / len(expected_sources)

    def run_evaluation(self, k: int = 10) -> None:
        """
        Exécute l'évaluation globale sur l'ensemble du jeu de données
        et affiche les statistiques détaillées en console.

        La méthode charge les données de résultats générés et le jeu
        de données de référence à partir de leurs chemins JSON respectifs.
        Elle filtre les questions pour conserver uniquement celles qui
        possèdent des sources attendues vérifiables.
        Ensuite, elle calcule et imprime le rappel (Recall) moyen sur toutes
        les questions évaluables pour les valeurs de k fixées à 1, 3, 5 et 10.

        Args:
            k (int): La valeur maximale de 'k' pour laquelle les métriques de
                     rappel doivent être calculées (défaut: 10).

        Returns:
            None
        """
        try:
            with open(self.student_search_results_path, 'r',
                      encoding='utf-8') as f:
                student_data = json.load(f)

            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)

            student_questions = student_data.get("search_results", [])
            truth_questions = ground_truth.get("rag_questions", [])

            print("Student data is valid: True")
            print(f"Total number of questions: {len(truth_questions)}")

            evaluable_questions = [q for q in truth_questions
                                   if q.get("sources")]
            print("Total number of questions with sources: "
                  f"{len(evaluable_questions)}")
            print(f"Total number of questions with student sources: "
                  f"{len(student_questions)}\n")
            print("Evaluation Results")
            print(f"Questions evaluated: {len(evaluable_questions)}")

            k_values = [1, 3, 5, 10]
            for current_k in k_values:
                if current_k > k:
                    continue

                total_recall = 0.0

                for truth_q in evaluable_questions:
                    q_id = truth_q.get("question_id")
                    expected_sources = truth_q.get("sources", [])

                    student_res = next((item for item in student_questions
                                        if item.get("question_id") == q_id),
                                       None)

                    if student_res:
                        retrieved_sources = student_res.get(
                            "retrieved_sources", [])
                        recall = self._evaluate_question_at_k(
                            retrieved_sources, expected_sources, current_k)
                        total_recall += recall

                avg_recall = total_recall / len(
                    evaluable_questions) if evaluable_questions else 0.0
                print(f"Recall@{current_k:02d}: {avg_recall:.3f}")

        except Exception as e:
            print(f"Erreur lors de l'évaluation : {e}")
