from __future__ import annotations
import os
import asyncio
import json
import random
import time
from typing import List
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
from tqdm.asyncio import tqdm

from .types import DataExample, Candidate, PydanticRng
from .adapter import Adapter, EvaluationBatch
from .pareto import add_candidate, ParetoFrontier, select_candidate_from_pareto_front
from .sampler import InfiniteIdxSampler
from .persistence import save_state, load_state

class OptimizationState(BaseModel):
    sampler: InfiniteIdxSampler
    frontier: ParetoFrontier = ParetoFrontier(entries=[])
    iteration: int = 0
    rng: PydanticRng = random.Random(0)
    model_config = ConfigDict(extra='allow', validate_assignment=True)

@dataclass
class WorkReservation:
    iteration_num: int
    parent_candidate: Candidate
    parent_candidate_idx: int
    minibatch: List[DataExample]
    component_name: str

def _reserve_work(
    s: OptimizationState, 
    components: List[str],
    trainset: List[DataExample]
    ) -> WorkReservation:

    s.iteration += 1
    iteration_num = s.iteration
    component_name = components[iteration_num % len(components)]
    candidate_idx = select_candidate_from_pareto_front(s.frontier, s.rng)
    candidate = s.frontier.entries[candidate_idx].candidate
    minibatch = _sample_minibatch(s.sampler, trainset)
    return WorkReservation(
        iteration_num, 
        candidate, 
        candidate_idx,
        minibatch, 
        component_name
    )

def _sample_minibatch(sampler: InfiniteIdxSampler, trainset: List[DataExample]) -> List[DataExample]:
    return [trainset[i] for i in sampler.next_minibatch_indices()]

def accept(init_eval: EvaluationBatch, second_eval: EvaluationBatch) -> bool:
    return second_eval.mean_score() > init_eval.mean_score()

async def optimize(
    adapter: Adapter,
    trainset: List[DataExample],
    valset: List[DataExample],
    seed_candidate: Candidate,
    train_attempts: int = 1,
    val_attempts: int = 1,
    lanes: int = 1,
    minibatch_size: int = 3,
    max_iterations: int = 200,
    perfect_score: float = 1.0,
    skip_perfect_score: bool = True,
    run_dir: str = 'run'
):
    s = load_state(run_dir)
    if s is None:
        s = OptimizationState(
            sampler=InfiniteIdxSampler(minibatch_size=minibatch_size, trainset_size=len(trainset))
        )
        start = time.time()
        initial_eval = await adapter.evaluate(valset, seed_candidate, attempts=val_attempts)
        print(f'initial_eval score: {initial_eval.mean_score()} in {time.time() - start} seconds')
        s.frontier = add_candidate(s.frontier, seed_candidate, initial_eval)
        save_state(run_dir, s)

    progress = tqdm(total=max_iterations, desc='Optimization Iteration')
    progress.display()

    async def _lane_loop(lane_id: int):
        components = adapter.components()
        while s.iteration < max_iterations:
            w = _reserve_work(s, components, trainset)
            init_eval = await adapter.evaluate(w.minibatch, w.parent_candidate, attempts=train_attempts)

            if skip_perfect_score and init_eval.mean_score() >= perfect_score:
                progress.update(1)
                print(f'skipping perfect score {init_eval.mean_score()} at iteration {w.iteration_num}')
                continue
            
            new_candidate = await adapter.propose_new_texts(w.parent_candidate, init_eval, w.component_name)
            second_eval = await adapter.evaluate(w.minibatch, new_candidate, attempts=train_attempts)

            if not accept(init_eval, second_eval):
                progress.update(1)
                print(f'not accepted with score {second_eval.mean_score()} vs {init_eval.mean_score()} at iteration {w.iteration_num}')
                continue

            print(f'accepted with score {second_eval.mean_score()} vs {init_eval.mean_score()} at iteration {w.iteration_num}')
            full_eval = await adapter.evaluate(valset, new_candidate, attempts=val_attempts)
            print(f'new candidate got full score {full_eval.mean_score()} at iteration {w.iteration_num}')
            s.frontier = add_candidate(s.frontier, new_candidate, full_eval)
            save_state(run_dir, s)
            progress.update(1)

    await asyncio.gather(*(_lane_loop(i) for i in range(lanes)))


