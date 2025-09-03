from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional, Callable


def ensure_run_dir(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)


# -----------------------------
# RNG state (tuple <-> JSON)
# -----------------------------


def _tuplify(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_tuplify(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _tuplify(v) for k, v in obj.items()}
    return obj


def _listify(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_listify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _listify(v) for k, v in obj.items()}
    return obj


def rng_state_to_json(rng: random.Random) -> Any:
    state = rng.getstate()
    return _listify(state)


def rng_state_from_json(obj: Any, rng: random.Random) -> None:
    state = _tuplify(obj)
    rng.setstate(state)  # type: ignore[arg-type]


# -----------------------------
# OptimizationState (JSON)
# -----------------------------


def serialize_state(state: Any) -> Dict[str, Any]:
    return {
        "candidates": list(state.candidates),
        "candidate_val_scores": list(state.candidate_val_scores),
        "candidate_val_subscores": [list(x) for x in state.candidate_val_subscores],
        "pareto_front_scores_by_task": list(state.pareto_front_scores_by_task),
        "pareto_front_candidates_by_task": [
            list(s) for s in state.pareto_front_candidates_by_task
        ],
        "i": int(state.i),
        "num_full_ds_evals": int(state.num_full_ds_evals),
        "total_num_evals": int(state.total_num_evals),
        "num_metric_calls_by_discovery": list(state.num_metric_calls_by_discovery),
    }


def deserialize_state(d: Dict[str, Any]) -> Any:
    # Lazy import to avoid circular import at module load
    from .core import OptimizationState  # type: ignore

    return OptimizationState(
        candidates=list(d.get("candidates") or []),
        candidate_val_scores=list(d.get("candidate_val_scores") or []),
        candidate_val_subscores=[
            list(x) for x in d.get("candidate_val_subscores") or []
        ],
        pareto_front_scores_by_task=list(d.get("pareto_front_scores_by_task") or []),
        pareto_front_candidates_by_task=[
            set(s) for s in d.get("pareto_front_candidates_by_task") or []
        ],
        i=int(d.get("i", -1)),
        num_full_ds_evals=int(d.get("num_full_ds_evals", 0)),
        total_num_evals=int(d.get("total_num_evals", 0)),
        num_metric_calls_by_discovery=list(
            d.get("num_metric_calls_by_discovery") or []
        ),
    )


# -----------------------------
# Sampler state passthrough
# -----------------------------


def resume_checkpoint(
    run_dir: str,
    *,
    rng: random.Random,
    sampler: Any,
) -> Optional[Any]:
    data = load_checkpoint(run_dir)
    if data is None:
        return None
    state = deserialize_state(data.get("state") or {})
    rng_state_obj = data.get("rng_state")
    if rng_state_obj is not None:
        rng_state_from_json(rng_state_obj, rng)
    sampler_state = data.get("sampler") or {}
    sampler.load_state_dict(sampler_state)
    return state


# -----------------------------
# Checkpoint and config I/O
# -----------------------------


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_checkpoint(
    run_dir: str,
    *,
    state: Any,
    rng: random.Random,
    sampler: Any,
) -> None:
    ensure_run_dir(run_dir)
    payload = {
        "version": 1,
        "state": serialize_state(state),
        "rng_state": rng_state_to_json(rng),
        "sampler": sampler.state_dict(),
        "last_iteration_completed": int(getattr(state, "i", -1)),
    }
    _atomic_write_json(os.path.join(run_dir, "checkpoint.json"), payload)


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_run_config(run_dir: str, config: Dict[str, Any]) -> None:
    ensure_run_dir(run_dir)
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        return
    _atomic_write_json(path, config)


def write_best_snapshot(run_dir: str, state: Any) -> None:
    if not (
        getattr(state, "candidate_val_scores", None)
        and getattr(state, "candidates", None)
    ):
        return
    scores = list(state.candidate_val_scores)
    best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    payload = {
        "best_index": int(best_idx),
        "best_score": float(scores[best_idx]) if scores else 0.0,
        "candidate": state.candidates[best_idx] if state.candidates else {},
        "num_candidates": len(state.candidates) if state.candidates else 0,
        "iteration": int(getattr(state, "i", -1)),
        "total_num_evals": int(getattr(state, "total_num_evals", 0)),
    }
    _atomic_write_json(os.path.join(run_dir, "best.json"), payload)


def save_checkpoint_and_best(
    run_dir: str,
    *,
    state: Any,
    rng: random.Random,
    sampler: Any,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    save_checkpoint(run_dir, state=state, rng=rng, sampler=sampler)
    write_best_snapshot(run_dir, state)
    if log is not None:
        num_candidates = len(getattr(state, "candidates", []) or [])
        total_evals = int(getattr(state, "total_num_evals", 0))
        log(
            f"✅ Checkpoint saved: iteration={state.i}, candidates={num_candidates}, total_evals={total_evals}"
        )
