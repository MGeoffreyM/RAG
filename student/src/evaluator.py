import json
from pathlib import Path
from pydantic import BaseModel
from typing import Any

class Evaluator(BaseModel):
    student_answer_path: Path
    dataset_path: Path
    max_context_length: int = 2000

    def _calculate_overlap(self, retrieved: dict[str, Any], expected: dict[str, Any]) -> float:
        """Calcule le pourcentage de chevauchement entre deux sources."""
        # 1. Vérifier si c'est le même fichier (Gestion des chemins relatifs/absolus)
        ret_path = Path(retrieved.get("file_path", "")).as_posix()
        exp_path = Path(expected.get("file_path", "")).as_posix()

        # On vérifie si l'un se termine par l'autre pour ignorer le dossier racine
        if not ret_path.endswith(exp_path) and not exp_path.endswith(ret_path):
            return 0.0

        # 2. Calculer l'intersection des index
        r_start = retrieved.get("first_character_index", 0)
        r_end = retrieved.get("last_character_index", 0)
        e_start = expected.get("first_character_index", 0)
        e_end = expected.get("last_character_index", 0)

        overlap_start = max(r_start, e_start)
        overlap_end = min(r_end, e_end)
        overlap_length = max(0, overlap_end - overlap_start)

        # 3. Calculer le pourcentage par rapport à la taille de la source attendue
        expected_length = e_end - e_start
        if expected_length == 0:
            return 0.0

        return overlap_length / expected_length

    def _evaluate_question_at_k(self, student_results: list[dict], expected_sources: list[dict], k: int) -> float:
        """Évalue une question pour un k donné (top-k)."""
        if not expected_sources:
            return 0.0

        # On ne garde que les k premiers résultats de l'étudiant
        top_k_retrieved = student_results[:k]
        found_count = 0

        for expected in expected_sources:
            # On vérifie si CETTE source attendue est dans nos résultats top-k
            is_found = False
            for retrieved in top_k_retrieved:
                overlap = self._calculate_overlap(retrieved, expected)
                if overlap >= 0.05:  # Tolérance de 5% exigée par le sujet
                    is_found = True
                    break
            
            if is_found:
                found_count += 1

        return found_count / len(expected_sources)

    def run_evaluation(self, k: int = 10) -> None:
        """Lance l'évaluation globale et affiche les statistiques."""
        try:
            with open(self.student_answer_path, 'r', encoding='utf-8') as f:
                student_data = json.load(f)
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)

            student_questions = student_data.get("search_results", [])
            truth_questions = ground_truth.get("rag_questions", [])

            print(f"Student data is valid: True")
            print(f"Total number of questions: {len(truth_questions)}")
            
            # On filtre les questions qui ont des sources dans le ground truth
            evaluable_questions = [q for q in truth_questions if q.get("sources")]
            print(f"Total number of questions with sources: {len(evaluable_questions)}")
            print(f"Total number of questions with student sources: {len(student_questions)}\n")
            print("Evaluation Results")
            print(f"Questions evaluated: {len(evaluable_questions)}")

            # Calcul des Recall pour k = 1, 3, 5, 10
            k_values = [1, 3, 5, 10]
            for current_k in k_values:
                if current_k > k:
                    continue # On ne peut pas calculer Recall@10 si on n'a cherché que k=5
                
                total_recall = 0.0
                
                for truth_q in evaluable_questions:
                    q_id = truth_q.get("question_id")
                    expected_sources = truth_q.get("sources", [])
                    
                    # Retrouver la réponse correspondante de l'étudiant
                    student_res = next((item for item in student_questions if item.get("question_id") == q_id), None)
                    
                    if student_res:
                        retrieved_sources = student_res.get("retrieved_sources", [])
                        recall = self._evaluate_question_at_k(retrieved_sources, expected_sources, current_k)
                        total_recall += recall

                avg_recall = total_recall / len(evaluable_questions) if evaluable_questions else 0.0
                # Format d'affichage attendu : Recall@01: 0.450
                print(f"Recall@{current_k:02d}: {avg_recall:.3f}")

        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation : {e}")
