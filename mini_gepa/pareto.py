from __future__ import annotations

import random
from typing import Dict, List, Set


def _is_dominated(
    y: int,
    other_programs: Set[int],
    program_at_pareto_front_valset: List[Set[int]],
) -> bool:
    """Return True if candidate y is dominated across all fronts.

    A candidate y is considered dominated if, for every front that contains y,
    there exists at least one other program from other_programs also in that front.
    """
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in other_programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    return True


def _remove_dominated_programs(
    program_at_pareto_front_valset: List[Set[int]],
    scores: List[float] | Dict[int, float] | None,
) -> List[Set[int]]:
    """Filter out dominated programs from all fronts.

    Mirrors the simplified GEPA behavior used for candidate selection.
    """
    # Count frequency of appearances across fronts to determine the candidate set
    freq: Dict[int, int] = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated: Set[int] = set()
    programs: List[int] = list(freq.keys())

    if scores is None:
        score_map: Dict[int, float] = {p: 1.0 for p in programs}
    elif isinstance(scores, dict):
        score_map = scores  # assume provides all needed indices
    else:
        score_map = {i: scores[i] for i in programs}

    # Sort from lowest to highest score to eliminate weaker programs first
    programs_sorted = sorted(programs, key=lambda x: score_map[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        remaining: Set[int] = set(programs_sorted).difference(dominated)
        for y in programs_sorted:
            if y in dominated:
                continue
            others = remaining.difference({y})
            if _is_dominated(y, others, program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs_sorted if p not in dominated]

    # Keep only dominators in each front
    new_program_at_pareto_front_valset = [
        {prog_idx for prog_idx in front if prog_idx in dominators}
        for front in program_at_pareto_front_valset
    ]
    return new_program_at_pareto_front_valset


def select_candidate_from_pareto_front(
    program_at_pareto_front_valset: List[Set[int]],
    agg_scores: List[float],
    rng: random.Random,
) -> int:
    """Frequency-weighted sampling over non-dominated Pareto fronts.

    1) Remove dominated programs (based on fronts + scores ordering)
    2) Count frequency across remaining fronts
    3) Sample proportional to frequency
    """
    pruned_fronts = _remove_dominated_programs(
        program_at_pareto_front_valset, agg_scores
    )

    freq: Dict[int, int] = {}
    for front in pruned_fronts:
        for prog_idx in front:
            freq[prog_idx] = freq.get(prog_idx, 0) + 1

    sampling_list = [prog_idx for prog_idx, count in freq.items() for _ in range(count)]
    assert len(sampling_list) > 0
    return rng.choice(sampling_list)
