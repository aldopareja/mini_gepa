from __future__ import annotations

import random
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, List, Optional, Tuple

from .pareto import select_candidate_from_pareto_front
from .sampler import EpochShuffledBatchSampler
from .types import CandidateProposal, DataInst, EvaluationBatch, RolloutOutput
from .persistence import resume_checkpoint, save_checkpoint_and_best


# ----------------------------------------
# Simple helpers
# ----------------------------------------


def get_head_and_tail(
    text: str, head_chars: int = 100, tail_chars: int = 100
) -> Tuple[str, str]:
    head = text[: max(0, head_chars)] if isinstance(text, str) else ""
    tail = (
        text[-max(0, tail_chars) :]
        if isinstance(text, str) and len(text) > tail_chars
        else ""
    )
    return head, tail


def get_candidate_head_and_tail(candidate: dict[str, str]) -> Tuple[str, str]:
    stringified = json.dumps(candidate, indent=2)
    return get_head_and_tail(stringified)


# ----------------------------------------
# Public dataclasses and results
# ----------------------------------------


@dataclass
class OptimizationResult:
    best_score: float
    best_program: dict[str, str]
    state: Any


# ----------------------------------------
# Optimization state and helpers
# ----------------------------------------


@dataclass
class OptimizationState:
    candidates: List[dict[str, str]]
    candidate_val_scores: List[float]
    candidate_val_subscores: List[List[float]]
    pareto_front_scores_by_task: List[float]
    pareto_front_candidates_by_task: List[set[int]]
    i: int
    num_full_ds_evals: int
    total_num_evals: int
    num_metric_calls_by_discovery: List[int]

    def update_with_new_candidate(
        self,
        *,
        parent_candidate_idx: List[int],
        new_candidate: dict[str, str],
        valset_outputs: List[RolloutOutput],
        valset_subscores: List[float],
        run_dir: Optional[str],
        num_metric_calls_by_discovery_of_new_candidate: int,
    ) -> None:
        new_candidate_idx = len(self.candidates)
        self.candidates.append(new_candidate)
        self.num_metric_calls_by_discovery.append(
            num_metric_calls_by_discovery_of_new_candidate
        )

        mean_score = (
            sum(valset_subscores) / len(valset_subscores) if valset_subscores else 0.0
        )
        self.candidate_val_scores.append(mean_score)

        # Track per-task scores
        self.candidate_val_subscores.append(list(valset_subscores))

        for task_idx, (old_score, new_score) in enumerate(
            zip(self.pareto_front_scores_by_task, valset_subscores, strict=False)
        ):
            if new_score > old_score:
                self.pareto_front_scores_by_task[task_idx] = new_score
                self.pareto_front_candidates_by_task[task_idx] = {new_candidate_idx}
            elif new_score == old_score:
                self.pareto_front_candidates_by_task[task_idx].add(new_candidate_idx)


async def initialize_state(
    *,
    run_dir: Optional[str],
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[
        [dict[str, str]], Awaitable[Tuple[List[RolloutOutput], List[float]]]
    ],
) -> OptimizationState:
    _outputs, subscores = await valset_evaluator(seed_candidate)
    base_score = sum(subscores) / len(subscores) if subscores else 0.0

    candidates: List[dict[str, str]] = [seed_candidate]
    candidate_val_scores: List[float] = [base_score]
    candidate_val_subscores: List[List[float]] = [list(subscores)]
    pareto_front_scores_by_task: List[float] = list(subscores)
    pareto_front_candidates_by_task: List[set[int]] = [{0} for _ in subscores]

    return OptimizationState(
        candidates=candidates,
        candidate_val_scores=candidate_val_scores,
        candidate_val_subscores=candidate_val_subscores,
        pareto_front_scores_by_task=pareto_front_scores_by_task,
        pareto_front_candidates_by_task=pareto_front_candidates_by_task,
        i=-1,
        num_full_ds_evals=1,
        total_num_evals=len(subscores),
        num_metric_calls_by_discovery=[0],
    )


# ----------------------------------------
# Proposal step (reflective mutation)
# ----------------------------------------


async def propose_once(
    *,
    curr_prog_id: int,
    curr_prog: dict[str, str],
    minibatch: List[DataInst],
    adapter: Any,
    components_to_update: List[str],
    perfect_score: float,
    skip_perfect_score: bool,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[CandidateProposal], bool]:
    """Run one propose-evaluate step on a provided minibatch.

    Returns (proposal_or_none, did_run_second_eval).
    """
    log = log or print

    # 1) Evaluate current program with traces
    eval_curr: EvaluationBatch = await adapter.evaluate(
        minibatch, curr_prog, capture_traces=True
    )

    if (
        skip_perfect_score
        and eval_curr.scores
        and all(s >= perfect_score for s in eval_curr.scores)
    ):
        log("All subsample scores perfect. Skipping mutation.")
        return None, False

    # 2) Propose new text for selected component(s)
    reflective_dataset = adapter.make_reflective_dataset(
        curr_prog, eval_curr, components_to_update
    )
    new_texts = await adapter.propose_new_texts(
        curr_prog, reflective_dataset, components_to_update
    )

    new_candidate = curr_prog.copy()
    for name, text in new_texts.items():
        new_candidate[name] = text

    # Log the transformation old -> new
    oh, ot = get_candidate_head_and_tail(curr_prog)
    log(f"🔄 candidate change: [{oh}'...'{ot}]\n ->\n {json.dumps(new_candidate, indent=2)}")

    # 3) Evaluate the new candidate (no traces needed)
    eval_new: EvaluationBatch = await adapter.evaluate(
        minibatch, new_candidate, capture_traces=False
    )

    # Acceptance test: compare means to keep semantics consistent with val
    old_scores = list(eval_curr.scores or [])
    new_scores = list(eval_new.scores or [])
    old_mean = (sum(old_scores) / len(old_scores)) if old_scores else 0.0
    new_mean = (sum(new_scores) / len(new_scores)) if new_scores else 0.0
    if new_mean <= old_mean:
        log(f"New subsample mean not better ({new_mean} <= {old_mean}). Skipping.")
        return None, True

    return (
        CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=list(range(len(minibatch))),  # placeholder indices
            subsample_scores_before=list(eval_curr.scores or []),
            subsample_scores_after=list(eval_new.scores or []),
            tag="reflective_mutation",
        ),
        True,
    )


# ----------------------------------------
# Engine and public optimize()
# ----------------------------------------


class Optimizer:
    def __init__(
        self,
        *,
        run_dir: Optional[str],
        evaluator: Callable[
            [List[DataInst], dict[str, str]], Tuple[List[RolloutOutput], List[float]]
        ],
        valset: List[DataInst],
        trainset: List[DataInst],
        seed_candidate: dict[str, str],
        max_metric_calls: int,
        perfect_score: float,
        adapter: Any,
        sampler: EpochShuffledBatchSampler,
        rng: random.Random,
        skip_perfect_score: bool = True,
        raise_on_exception: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.run_dir = run_dir
        self.evaluator = evaluator
        self.valset = valset
        self.trainset = trainset
        self.seed_candidate = seed_candidate
        self.max_metric_calls = max_metric_calls
        self.perfect_score = perfect_score
        self.adapter = adapter
        self.sampler = sampler
        self.rng = rng
        self.skip_perfect_score = skip_perfect_score
        self.raise_on_exception = raise_on_exception
        self.log = log or print

        # Require adapter to declare the components to optimize
        if not hasattr(self.adapter, "components"):
            raise ValueError("Adapter must implement components() -> List[str]")
        components = self.adapter.components()
        if not isinstance(components, list) or not all(
            isinstance(c, str) and c for c in components
        ):
            raise ValueError("adapter.components() must return a non-empty List[str]")
        self._components: List[str] = components
        self._rr_idx: int = 0

    def _val_evaluator(
        self,
    ) -> Callable[[dict[str, str]], Awaitable[Tuple[List[RolloutOutput], List[float]]]]:
        return lambda prog: self.evaluator(self.valset, prog)

    async def _run_full_eval_and_add(
        self,
        *,
        new_program: dict[str, str],
        state: OptimizationState,
        parent_program_idx: List[int],
    ) -> None:
        num_metric_calls_by_discovery = state.total_num_evals
        vh, vt = get_candidate_head_and_tail(new_program)
        self.log(f"Validation eval: head='{vh}' tail='{vt}'")
        valset_outputs, valset_subscores = await self._val_evaluator()(new_program)
        # Atomic state update (no awaits)
        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)
        state.update_with_new_candidate(
            parent_candidate_idx=parent_program_idx,
            new_candidate=new_program,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_candidate=num_metric_calls_by_discovery,
        )

    async def run(self, lanes: int) -> OptimizationState:
        # Attempt resume from checkpoint if available
        state: OptimizationState
        resumed = False
        if self.run_dir:
            maybe_state = resume_checkpoint(
                self.run_dir, rng=self.rng, sampler=self.sampler
            )
            if maybe_state is not None:
                state = maybe_state
                self.log(
                    f"Resuming from checkpoint: iteration={state.i}, total_evals={state.total_num_evals}"
                )
                resumed = True

        if not resumed:
            # Log seed validation evaluation
            sh, st = get_candidate_head_and_tail(self.seed_candidate)
            self.log(f"Validation eval (seed): head='{sh}' tail='{st}'")

            state = await initialize_state(
                run_dir=self.run_dir,
                seed_candidate=self.seed_candidate,
                valset_evaluator=self._val_evaluator(),
            )

            self.log(
                f"Iteration {state.i + 1}: Base candidate full valset score: "
                f"{state.candidate_val_scores[0]}"
            )

        # Helpers for atomic reservation and minibatch selection
        def reserve_iteration_and_prepare() -> (
            Tuple[int, int, dict[str, str], List[int], List[DataInst]]
        ):
            iteration_num = state.i + 1
            # Select parent program id from Pareto
            curr_prog_id = select_candidate_from_pareto_front(
                state.pareto_front_candidates_by_task,
                state.candidate_val_scores,
                self.rng,
            )
            curr_prog = state.candidates[curr_prog_id]
            # Update iteration counter
            state.i = iteration_num
            # Subsample minibatch deterministically based on iteration
            subsample_ids = self.sampler.next_minibatch_indices(
                len(self.trainset), iteration_num - 1
            )
            minibatch = [self.trainset[j] for j in subsample_ids]
            self.log(
                f"Iteration {iteration_num}: Selected program {curr_prog_id} score: {state.candidate_val_scores[curr_prog_id]}"
            )
            return iteration_num, curr_prog_id, curr_prog, subsample_ids, minibatch

        async def lane_loop(lane_id: int) -> None:
            while state.total_num_evals < self.max_metric_calls:
                try:
                    iteration_num, curr_prog_id, curr_prog, subsample_ids, minibatch = (
                        reserve_iteration_and_prepare()
                    )
                    # Round-robin selection of component to update
                    component_name = self._components[self._rr_idx % len(self._components)]
                    components_to_update = [component_name]
                    self._rr_idx += 1

                    proposal, did_second_eval = await propose_once(
                        curr_prog_id=(
                            curry_prog_id
                            if (curry_prog_id := curr_prog_id) is not None
                            else curr_prog_id
                        ),
                        curr_prog=curr_prog,
                        minibatch=minibatch,
                        adapter=self.adapter,
                        components_to_update=components_to_update,
                        perfect_score=self.perfect_score,
                        skip_perfect_score=self.skip_perfect_score,
                        log=lambda s: self.log(f"[lane {lane_id}] {s}"),
                    )

                    # Atomic update of eval counters
                    state.total_num_evals += len(subsample_ids)
                    if did_second_eval:
                        state.total_num_evals += len(subsample_ids)

                    if proposal is None:
                        continue

                    await self._run_full_eval_and_add(
                        new_program=proposal.candidate,
                        state=state,
                        parent_program_idx=proposal.parent_program_ids,
                    )
                except Exception as e:
                    self.log(f"[lane {lane_id}] Exception during optimization: {e}")
                    if self.raise_on_exception:
                        raise
                    else:
                        continue
                finally:
                    save_checkpoint_and_best(
                        self.run_dir,
                        state=state,
                        rng=self.rng,
                        sampler=self.sampler,
                        log=self.log,
                    )

        # Launch lanes
        lanes = max(1, int(lanes))
        await asyncio.gather(*(lane_loop(i) for i in range(lanes)))

        return state


async def optimize(
    *,
    seed_candidate: dict[str, str],
    trainset: List[DataInst],
    valset: List[DataInst],
    adapter: Any,
    skip_perfect_score: bool = True,
    minibatch_size: int = 3,
    lanes: int = 1,
    perfect_score: float = 1.0,
    max_metric_calls: int,
    run_dir: Optional[str] = None,
    seed: int = 0,
    raise_on_exception: bool = True,
) -> OptimizationResult:
    rng = random.Random(seed)

    async def evaluator(
        inputs: List[DataInst], prog: dict[str, str]
    ) -> Tuple[List[RolloutOutput], List[float]]:
        eval_out = await adapter.evaluate(
            inputs, prog, capture_traces=False, attempts=adapter.val_attempts
        )
        return eval_out.outputs, eval_out.scores

    sampler = EpochShuffledBatchSampler(minibatch_size=minibatch_size, rng=rng)

    engine = Optimizer(
        run_dir=run_dir,
        evaluator=evaluator,
        valset=valset,
        trainset=trainset,
        seed_candidate=seed_candidate,
        max_metric_calls=max_metric_calls,
        perfect_score=perfect_score,
        adapter=adapter,
        sampler=sampler,
        rng=rng,
        skip_perfect_score=skip_perfect_score,
        raise_on_exception=raise_on_exception,
    )

    state = await engine.run(lanes=lanes)
    best_idx = max(
        range(len(state.candidate_val_scores)),
        key=lambda i: state.candidate_val_scores[i],
    )
    return OptimizationResult(
        best_score=state.candidate_val_scores[best_idx],
        best_program=state.candidates[best_idx],
        state=state,
    )
