from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict

from .types import DataExample

ComponentName = str
Candidate = Dict[ComponentName, Any]


@dataclass
class AttemptEvalOutput:
    task_idx: int
    attempt_idx: int
    rollout_output: str
    trace: Dict[str, Any] # it's an arbitrary dict with fields like "Inputs", "Generated Outputs", "Expected_Answer_String", etc.
    score: float
    candidate: Candidate

@dataclass
class EvaluationBatch:
    batch_results: List[AttemptEvalOutput]

    def mean_score(self) -> float:
        num_attempts = len(self.batch_results)
        total_score = sum(r.score for r in self.batch_results)
        return total_score / num_attempts

    def max_score_per_task(self) -> Dict[int, float]:
        def _max_score_per_task(task_idx: int) -> float:
            return max(r.score for r in self.batch_results if r.task_idx == task_idx)
        return {task_idx: _max_score_per_task(task_idx) for task_idx in self.get_unique_tasks()}
    
    def mean_score_per_task(self) -> Dict[int, float]:
        def _mean_score_per_task(task_idx: int) -> float:
            task_attempts = [r for r in self.batch_results if r.task_idx == task_idx]
            return sum(r.score for r in task_attempts) / len(task_attempts)
        return {task_idx: _mean_score_per_task(task_idx) for task_idx in self.get_unique_tasks()}

    def get_unique_tasks(self) -> List[int]:
        return sorted(list(set(r.task_idx for r in self.batch_results)))

class Adapter(BaseModel):
    model_config = ConfigDict(extra='allow', validate_assignment=True)

    def components(self) -> List[ComponentName]:
        """Return the list of component names that can be optimized.

        Example: ["system_prompt"].
        """
        pass

    async def evaluate(self, batch: List[DataExample], candidate: Candidate, attempts: int, **kwargs) -> EvaluationBatch:
        pass

    async def propose_new_texts(self, candidate: Candidate, eval_batch: EvaluationBatch, component_to_update: ComponentName, **kwargs) -> Candidate:
        pass


