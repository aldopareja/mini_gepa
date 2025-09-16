from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import typer

from datasets import load_dataset

from mini_gepa.core import optimize
from mini_gepa.types import EvaluationBatch
from mini_gepa.persistence import write_run_config
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
    train_attempts: int = 1
    val_attempts: int = 1
    max_concurrency: int = 32
    reasoning_effort: Optional[str] = None  # e.g., "high" for models that support it

    # Components API
    def components(self) -> List[str]:
        return ["system_prompt"]

    async def _one_call(self, system_prompt: str, user_input: str) -> str:
        req: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        }
        if self.reasoning_effort is not None:
            req["reasoning"] = {"effort": self.reasoning_effort}

        resp = await responses_create(**req)
        return getattr(resp, "output_text", "") or ""

    async def _score_instance(self, example: Dict[str, Any], candidate: Dict[str, str], attempts: int, sem: asyncio.Semaphore) -> tuple[Dict[str, Any], float, Dict[str, Any], List[float]]:
        system_prompt = candidate.get("system_prompt", "")
        user_input = str(example["input"])
        expected = str(example["answer"])  # already "### <answer>"

        async def once() -> str:
            async with sem:
                return await self._one_call(system_prompt, user_input)

        # Run attempts fully async without bottlenecking the entire loop
        responses = await asyncio.gather(*(once() for _ in range(max(1, int(attempts)))))

        # Simple metric: 1.0 if expected substring appears in response, else 0.0; average over attempts
        attempt_scores = [1.0 if expected in r else 0.0 for r in responses]
        mean_score = sum(attempt_scores) / len(attempt_scores) if attempt_scores else 0.0

        # Keep last response for outputs
        output = {"full_assistant_response": '\n'.join(responses) if responses else ""}
        trace = {
            "Inputs": user_input,
            "Generated Outputs": '\n'.join(responses) if responses else "",
            "Expected_Answer_String": expected,
        }
        return output, mean_score, trace, attempt_scores

    async def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = False,
        attempts: Optional[int] = None,
    ) -> EvaluationBatch:
        att = attempts if attempts is not None else self.train_attempts
        sem = asyncio.Semaphore(self.max_concurrency)

        outs: List[Dict[str, Any]] = []
        scores: List[float] = []
        attempt_scores: List[List[float]] = []
        traces: Optional[List[Dict[str, Any]]] = [] if capture_traces else None

        results = await asyncio.gather(
            *(self._score_instance(ex, candidate, att, sem) for ex in batch)
        )
        for output, score, trace, per_attempt in results:
            outs.append(output)
            scores.append(score)
            attempt_scores.append(per_attempt)
            if capture_traces and traces is not None:
                traces.append(trace)

        return EvaluationBatch(outputs=outs, scores=scores, attempt_scores=attempt_scores, trajectories=traces)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Minimal reflective dataset using the outputs and any traces
        comp = components_to_update[0]
        records: List[Dict[str, Any]] = []
        outputs = eval_batch.outputs
        scores = eval_batch.scores
        traces = eval_batch.trajectories or [None] * len(outputs)

        for out, score, trace in zip(outputs, scores, traces, strict=False):
            gen = (out or {}).get("full_assistant_response", "")
            feedback = "Response includes correct answer." if score > 0.5 else "Response missing the exact required answer string."
            record = {
                "Generated Outputs": gen,
                "Feedback": feedback,
            }
            if isinstance(trace, dict):
                record.update({**trace})
            records.append(record)

        return {comp: records}

    async def propose_new_texts(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, List[Dict[str, Any]]],
        components_to_update: List[str],
    ) -> Dict[str, str]:
        # Very simple teacher prompt to rewrite system_prompt
        out: Dict[str, str] = {}
        for comp in components_to_update:
            examples = reflective_dataset.get(comp, [])
            current_text = candidate.get(comp, "")
            prompt = (
                "You are optimizing an instruction for solving AIME-style math problems.\n"
                "Your goal is to rewrite the current system prompt to improve the performance of the assistant.\n"
                "Current system prompt:\n" + current_text + "\n\n"
                "Here are some example model responses and feedback:\n" + json.dumps(examples[:10], indent=2) + "\n\n"
                "carefully consider the feedback and the model responses to improve the system prompt.\n"
                "Return only the improved system prompt text, with no extra commentary."
            )
            req: Dict[str, Any] = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
            }
            if self.reasoning_effort is not None:
                req["reasoning"] = {"effort": self.reasoning_effort}
            resp = await responses_create(**req)
            improved = resp.output_text
            out[comp] = improved
        return out


# -----------------------------
# CLI
# -----------------------------


@app.command()
def run(
    model: str = typer.Option("gpt-4.1-mini", help="OpenAI model for both actor and teacher"),
    reasoning_effort: Optional[str] = typer.Option(None, help="Reasoning effort to pass to the API (e.g., 'high')"),
    train_size: int = typer.Option(12, help="Num training instances (<= available)"),
    val_size: int = typer.Option(12, help="Num validation instances (<= available)"),
    train_attempts: int = typer.Option(1, help="Attempts per example during training minibatches"),
    val_attempts: int = typer.Option(1, help="Attempts per example during validation"),
    minibatch_size: int = typer.Option(3, help="Training minibatch size"),
    lanes: int = typer.Option(8, help="Number of concurrent optimizer lanes"),
    max_metric_calls: int = typer.Option(200, help="Total evaluation budget (minibatch elements counted)"),
    run_dir: str = typer.Option("runs/aime_minimal", help="Directory for checkpoints and artifacts"),
    seed: int = typer.Option(0, help="Random seed for optimizer"),
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
            train_attempts=train_attempts,
            val_attempts=val_attempts,
            max_concurrency=64,
            reasoning_effort=reasoning_effort,
        )

        write_run_config(
            run_dir,
            {
                "model": model,
                "train_size": train_size,
                "val_size": val_size,
                "train_attempts": train_attempts,
                "val_attempts": val_attempts,
                "minibatch_size": minibatch_size,
                "lanes": lanes,
                "max_metric_calls": max_metric_calls,
                "seed": seed,
                "reasoning_effort": reasoning_effort,
            },
        )

        result = await optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            skip_perfect_score=True,
            minibatch_size=minibatch_size,
            lanes=lanes,
            perfect_score=1.0,
            max_metric_calls=max_metric_calls,
            run_dir=run_dir,
            seed=seed,
            raise_on_exception=True,
        )

        typer.echo("Best score: " + str(result.best_score))
        typer.echo("Best program: \n" + json.dumps(result.best_program, indent=2))

    asyncio.run(_main())


if __name__ == "__main__":
    app()

