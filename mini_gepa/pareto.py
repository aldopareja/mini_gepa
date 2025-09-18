from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List
import random

from .types import Candidate
from .adapter import EvaluationBatch

CandidateIdx = int
TaskIdx = int

@dataclass
class ParetoFrontEntry:
    candidate: Candidate
    candidate_idx: CandidateIdx
    eval_results: EvaluationBatch

@dataclass
class ParetoFrontier:
    entries: List[ParetoFrontEntry]

def add_candidate(frontier: ParetoFrontier, candidate: Candidate, eval_results: EvaluationBatch) -> ParetoFrontier:
    """Add a candidate to the candidates dictionary."""
    candidates = frontier.entries
    candidate_idx = len(candidates)
    assert candidates[0].eval_results.get_unique_tasks() == eval_results.get_unique_tasks() if candidates else True
    candidates.append(ParetoFrontEntry(candidate, candidate_idx, eval_results))
    return frontier

ParetoMatrix = Dict[TaskIdx, Dict[CandidateIdx, float]]

def _get_pareto_matrix(frontier: ParetoFrontier) -> ParetoMatrix:
    """Compute a matrix where rows are task_ids, columns are candidate_ids, 
    and elements are the mean score of each candidate on each task."""
    
    # Get all task IDs from the first candidate
    first_candidate = frontier.entries[0]
    task_ids = first_candidate.eval_results.get_unique_tasks()
    
    # Initialize matrix as dict of dicts
    matrix = defaultdict(dict)
    for task_id in task_ids:
        for entry in frontier.entries:
            # Get max score for this candidate on this task
            matrix[task_id][entry.candidate_idx] = entry.eval_results.mean_score_per_task()[task_id]
    
    return matrix

AggScores = Dict[CandidateIdx, float]

def _get_candidate_mean_scores(pareto_matrix: ParetoMatrix) -> AggScores:
    """Compute mean scores across all tasks for each candidate."""
    candidate_indices = pareto_matrix[0].keys()
    task_indices = pareto_matrix.keys()
    
    agg_scores = {}
    for candidate_idx in candidate_indices:
        candidate_scores_across_tasks = [pareto_matrix[task_id][candidate_idx] for task_id in task_indices]
        agg_scores[candidate_idx] = mean(candidate_scores_across_tasks)
    
    return agg_scores
    
def _remove_dominated_candidates(pareto_matrix: ParetoMatrix, agg_scores: AggScores) -> ParetoMatrix:
    """
    Find candidates that are not dominated by any other candidate across all tasks.
    
    A candidate is considered a "dominator" (non-dominated) if there exists at least one task
    where it performs strictly better than all other candidates. In other words, a candidate
    is dominated if for every task, there exists another candidate that performs at least as good.
    """
    
    task_indices = pareto_matrix.keys()

    ordered_candidate_indices = sorted(agg_scores.keys(), key=lambda x: agg_scores[x], reverse=False)
    dominators = set(ordered_candidate_indices)
    
    def _dominates_in_task(task_id: TaskIdx, candidate_idx: CandidateIdx) -> bool:
        """
        A candidate dominates if it is the only one with the max score on that task
        """
        front = pareto_matrix[task_id]
        max_score = max(front.values())
        best_candidate_indices = list(filter(lambda candidate_idx: front[candidate_idx] == max_score, dominators))
        return len(best_candidate_indices) == 1 and candidate_idx in best_candidate_indices

    removed = True
    while removed:
        removed = False
        for candidate_idx in ordered_candidate_indices:
            if candidate_idx not in dominators:
                continue
            if not any(_dominates_in_task(task_id, candidate_idx) for task_id in task_indices):
                dominators.remove(candidate_idx)
                removed = True
    
    new_pareto_matrix = defaultdict(dict)
    for task_id in task_indices:
        for candidate_idx, score in pareto_matrix[task_id].items():
            if candidate_idx in dominators:
                new_pareto_matrix[task_id][candidate_idx] = score
    
    return new_pareto_matrix

def _sample_based_on_frequency(pareto_matrix: ParetoMatrix, rng: random.Random) -> int:
    
    def _wins_in_task(candidate_idx: CandidateIdx, task_id: TaskIdx) -> bool:
        front = pareto_matrix[task_id]
        candidate_score = front[candidate_idx]
        return candidate_score == max(front.values())    
    
    win_counts: Dict[CandidateIdx, int] = defaultdict(int)

    candidate_indices = pareto_matrix[0].keys()
    task_indices = pareto_matrix.keys()
    win_counts = {candidate_idx: sum(_wins_in_task(candidate_idx, task_id) for task_id in task_indices) for candidate_idx in candidate_indices}
    
    sampling_list = [candidate_idx for candidate_idx, count in win_counts.items() for _ in range(count)]
    assert len(sampling_list) > 0
    return rng.choice(sampling_list)


def select_candidate_from_pareto_front(frontier: ParetoFrontier, rng: random.Random) -> int:
    pareto_matrix = _get_pareto_matrix(frontier)
    agg_scores = _get_candidate_mean_scores(pareto_matrix)
    
    new_pareto_matrix = _remove_dominated_candidates(pareto_matrix, agg_scores)

    candidate_idx = _sample_based_on_frequency(new_pareto_matrix, rng)
    return candidate_idx
