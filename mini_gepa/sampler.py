from __future__ import annotations

import random
from collections import Counter
from typing import List


class EpochShuffledBatchSampler:
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

    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> List[int]:
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
