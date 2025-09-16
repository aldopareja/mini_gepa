from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Basic aliases for readability
DataInst = Dict[str, Any]
RolloutOutput = Dict[str, Any]


@dataclass
class EvaluationBatch:
    outputs: List[RolloutOutput]
    scores: List[float]
    attempt_scores: List[List[float]]
    trajectories: Optional[List[Dict[str, Any]]] = None


@dataclass
class CandidateProposal:
    candidate: Dict[str, str]
    parent_program_ids: List[int]
    subsample_indices: List[int]
    subsample_scores_before: List[float]
    subsample_scores_after: List[float]
    subsample_attempt_scores_before: List[List[float]]
    subsample_attempt_scores_after: List[List[float]]
    tag: str = "reflective_mutation"
