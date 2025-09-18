from __future__ import annotations

import asyncio
from itertools import chain
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import typer

from datasets import load_dataset

from mini_gepa.core import DataExample, optimize
from mini_gepa.adapter import Candidate, ComponentName, EvaluationBatch, AttemptEvalOutput
from mini_gepa.openai_async import responses_create


app = typer.Typer(help="Minimal GEPA-style local optimizer demo on AIME with a simple adapter.")


# -----------------------------
# Dataset loader (AIME)
# -----------------------------

def load_aime_splits(train_size: Optional[int] = None, val_size: Optional[int] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_split = [
        {
            "input": x["problem"],
            "additional_context": {"solution": x.get("solution", "")},
            "answer": "### " + str(x["answer"]),
        }
        for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    # deterministic split: second half as validation
    mid = len(train_split) // 2
    trainset = train_split[:mid]
    valset = train_split[mid:]

    if isinstance(train_size, int) and train_size > 0:
        trainset = trainset[:train_size]
    if isinstance(val_size, int) and val_size > 0:
        valset = valset[:val_size]

    return trainset, valset


# -----------------------------
# Minimal Adapter and Actor
# -----------------------------


@dataclass
class AIMEAdapter:
    model: str
    max_concurrency: int = 32
    reasoning_effort: Optional[str] = None  # e.g., "high" for models that support it

    def components(self) -> List[str]:
        return ["system_prompt"]

    async def _one_call(self, system_prompt: str, user_input: str, sem: asyncio.Semaphore) -> str:
        req: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        }
        if self.reasoning_effort is not None:
            req["reasoning"] = {"effort": self.reasoning_effort}
        async with sem:
            resp = await responses_create(**req)
            return getattr(resp, "output_text", "") or ""

    async def _score_instance(self, task_idx: int, example: DataExample, candidate: Candidate, attempts: int, sem: asyncio.Semaphore) -> List[AttemptEvalOutput]:
        system_prompt = candidate["system_prompt"]
        user_input = example["input"]
        expected = example["answer"]  # e.g. "### <answer>"
        
        calls = [self._one_call(system_prompt, user_input, sem) for _ in range(max(1, int(attempts)))]
        responses = await asyncio.gather(*calls)

        attempt_eval_outputs: List[AttemptEvalOutput] = []
        for i, r in enumerate(responses):
            score = 1.0 if expected in r else 0.0
            trace = {
                "Inputs": user_input,
                "Generated Outputs": r,
                "Score": score,
                "Expected_Answer_String": expected,
            }
            attempt_eval_outputs.append(
                AttemptEvalOutput(
                    task_idx=task_idx,
                    candidate=candidate,
                    attempt_idx=i,
                    rollout_output=r,
                    trace=trace,
                    score=score
                )
            )
        return attempt_eval_outputs

    async def evaluate(
        self,
        batch: List[DataExample],
        candidate: Candidate,
        attempts: int,
    ) -> EvaluationBatch:
        sem = asyncio.Semaphore(self.max_concurrency)

        tasks = [self._score_instance(idx, ex, candidate, attempts, sem) for idx, ex in enumerate(batch)]
        eval_outputs = await asyncio.gather(*tasks)

        return EvaluationBatch(
            batch_results=list(chain(*eval_outputs)), # flatten list of lists
        )

    async def propose_new_texts(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        component_to_update: ComponentName,
    ) -> Candidate:
        trajectories = [e.trace for e in eval_batch.batch_results]


        current_text = candidate.get(component_to_update, "")
        prompt = (
            "You are optimizing an instruction for solving AIME-style math problems.\n"
            "Your goal is to rewrite the current system prompt to improve the performance of the assistant.\n"
            f"Current system prompt:\n{current_text}\n\n"
            f"Here are some example model responses and feedback:\n{json.dumps(trajectories[:10], indent=2)}\n\n"
            "Carefully consider the feedback and the model responses to improve the system prompt.\n"
            "Return only the improved system prompt text, with no extra commentary."
        )
        req: Dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
        }
        if self.reasoning_effort is not None:
            req["reasoning"] = {"effort": self.reasoning_effort}
        resp = await responses_create(**req)
        improved = getattr(resp, "output_text", "") or ""
        new_candidate = dict(candidate)
        new_candidate[component_to_update] = improved
        return new_candidate


# -----------------------------
# CLI
# -----------------------------


@app.command()
def run(
    model: str = typer.Option("gpt-4.1-mini", help="OpenAI model for both actor and teacher"),
    reasoning_effort: Optional[str] = typer.Option(None, help="Reasoning effort to pass to the API (e.g., 'high')"),
    train_size: int = typer.Option(12, help="Num training instances (<= available)"),
    val_size: int = typer.Option(3, help="Num validation instances (<= available)"),
    train_attempts: int = typer.Option(1, help="Attempts per example during training minibatches"),
    val_attempts: int = typer.Option(1, help="Attempts per example during validation"),
    minibatch_size: int = typer.Option(3, help="Training minibatch size"),
    lanes: int = typer.Option(8, help="Number of concurrent optimizer lanes"),
    max_iterations: int = typer.Option(200, help="Max number of optimization iterations"),
    run_dir: str = typer.Option('run', help="Run directory"),
):
    """Run a minimal GEPA-style optimization on AIME with a simple adapter.

    Requirements:
    - OPENAI_API_KEY must be set in the environment.
    - Hugging Face datasets will download AIME splits on first run.
    """
    import nest_asyncio
    nest_asyncio.apply()
    async def _main() -> None:
        trainset, valset = load_aime_splits(train_size, val_size)

        # Seed candidate: a simple math instruction
        seed_candidate = {
            "system_prompt": (
                "You are a helpful assistant for math olympiad problems.\n"
                "Read the problem and provide a complete, correct solution.\n"
            )
        }

        adapter = AIMEAdapter(
            model=model,
            max_concurrency=64,
            reasoning_effort=reasoning_effort,
        )

        await optimize(
            adapter=adapter,
            trainset=trainset,
            valset=valset,
            seed_candidate=seed_candidate,
            train_attempts=train_attempts,
            val_attempts=val_attempts,
            lanes=lanes,
            minibatch_size=minibatch_size,
            max_iterations=max_iterations,
            perfect_score=1.0,
            skip_perfect_score=True,
            run_dir=run_dir,
        )

    asyncio.run(_main())


if __name__ == "__main__":
    app()

