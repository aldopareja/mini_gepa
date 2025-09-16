from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List
import math


class Sampler:
    def next_minibatch_indices(self, trainset_size: int, iteration: int, candidate_id: int) -> List[int]:
        raise NotImplementedError

    def record_attempts(self, candidate_id: int, example_indices: List[int], attempt_scores: List[List[float]]) -> None:
        # default no-op for stateless samplers
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None


class EpochShuffledBatchSampler(Sampler):
    """
    Shuffle ids once per epoch, then iterate in fixed-size minibatches.
    If needed, pad the epoch to be a multiple of minibatch size by repeating
    least-frequent ids so far. Uses provided rng for determinism.

    Epoch boundaries are inferred from the zero-based iteration index passed to
    next_minibatch_indices(). At each new epoch, the sampler shuffles all indices
    [0..trainset_size-1], updates frequency counts, and pads the total length to
    a multiple of minibatch size by repeating the current least-frequent ids.
    """

    def __init__(self, minibatch_size: int, rng: random.Random) -> None:
        self.minibatch_size = int(minibatch_size)
        self.rng = rng
        self.shuffled_ids: List[int] = []
        self.epoch: int = -1
        self.id_freqs: Counter[int] = Counter()

    def _update_shuffled(self, trainset_size: int) -> None:
        # Fresh shuffle for the new epoch
        self.shuffled_ids = list(range(trainset_size))
        self.rng.shuffle(self.shuffled_ids)
        for idx in self.shuffled_ids:
            self.id_freqs[idx] += 1

        # Pad to make total length a multiple of minibatch_size
        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                # Pick the least frequent id so far
                least_frequent_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(least_frequent_id)
                self.id_freqs[least_frequent_id] += 1

    def next_minibatch_indices(self, trainset_size: int, iteration: int, candidate_id: int) -> List[int]:
        # iteration is zero-based in caller
        base_idx = iteration * self.minibatch_size
        curr_epoch = (
            0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)
        )
        if curr_epoch > self.epoch:
            self.epoch = curr_epoch
            self._update_shuffled(trainset_size)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]

    # -----------------------------
    # Persistence helpers
    # -----------------------------

    def state_dict(self) -> dict:
        return {
            "minibatch_size": int(self.minibatch_size),
            "epoch": int(self.epoch),
            "shuffled_ids": list(self.shuffled_ids),
            "id_freqs": dict(self.id_freqs),
        }

    def load_state_dict(self, state: dict) -> None:
        self.minibatch_size = int(state["minibatch_size"])  # happy path
        self.epoch = int(state["epoch"])  # happy path
        self.shuffled_ids = list(state["shuffled_ids"])  # happy path
        freqs = dict(state["id_freqs"])  # happy path
        self.id_freqs = Counter({int(k): int(v) for k, v in freqs.items()})


@dataclass
class ExampleStats:
    visits: int = 0
    attempt_count: int = 0
    sum_scores: float = 0.0
    sum_sq_scores: float = 0.0

    def update_with_attempts(self, attempts: List[float]) -> None:
        n = len(attempts)
        self.visits += 1
        self.attempt_count += n
        self.sum_scores += sum(attempts)
        self.sum_sq_scores += sum(a * a for a in attempts)

    def variance(self) -> float:
        if self.attempt_count < 2:
            return 1.0
        mean = self.sum_scores / self.attempt_count
        return max(0.0, (self.sum_sq_scores / self.attempt_count) - (mean * mean))


@dataclass
class AdaptiveVarianceSampler(Sampler):
    minibatch_size: int
    rng: random.Random
    temperature: float = 1.0
    stats: Dict[int, Dict[int, ExampleStats]] = field(default_factory=dict)

    def record_attempts(self, candidate_id: int, example_indices: List[int], attempt_scores: List[List[float]]) -> None:
        cand = self.stats.setdefault(candidate_id, {})
        for idx, attempts in zip(example_indices, attempt_scores, strict=False):
            es = cand.setdefault(idx, ExampleStats())
            es.update_with_attempts(list(attempts))

    def next_minibatch_indices(self, trainset_size: int, iteration: int, candidate_id: int) -> List[int]:
        cand = self.stats.get(candidate_id, {})
        logits = [cand.get(j, ExampleStats()).variance() for j in range(trainset_size)]
        probs = _softmax(logits, self.temperature)
        return _sample_without_replacement(self.rng, probs, self.minibatch_size)

    def state_dict(self) -> dict:
        out: Dict[str, Any] = {
            "minibatch_size": int(self.minibatch_size),
            "temperature": float(self.temperature),
            "stats": {},
        }
        for cid, ex_map in self.stats.items():
            out["stats"][str(cid)] = {
                str(eid): {
                    "visits": s.visits,
                    "attempt_count": s.attempt_count,
                    "sum_scores": s.sum_scores,
                    "sum_sq_scores": s.sum_sq_scores,
                }
                for eid, s in ex_map.items()
            }
        return out

    def load_state_dict(self, state: dict) -> None:
        self.minibatch_size = int(state["minibatch_size"])  # happy path
        self.temperature = float(state["temperature"])  # happy path
        self.stats = {}
        stats = state.get("stats") or {}
        for cid, ex_map in stats.items():
            self.stats[int(cid)] = {}
            for eid, s in (ex_map or {}).items():
                self.stats[int(cid)][int(eid)] = ExampleStats(
                    visits=int(s["visits"]),
                    attempt_count=int(s["attempt_count"]),
                    sum_scores=float(s["sum_scores"]),
                    sum_sq_scores=float(s["sum_sq_scores"]),
                )


def _softmax(logits: List[float], temperature: float) -> List[float]:
    t = max(1e-6, float(temperature))
    scaled = [x / t for x in logits]
    m = max(scaled) if scaled else 0.0
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _sample_without_replacement(rng: random.Random, probs: List[float], k: int) -> List[int]:
    n = len(probs)
    chosen: List[int] = []
    weights = list(probs)
    for _ in range(min(k, n)):
        total = sum(weights) or 1.0
        r = rng.random() * total
        acc = 0.0
        pick = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                pick = i
                break
        chosen.append(pick)
        weights[pick] = 0.0
    return chosen
