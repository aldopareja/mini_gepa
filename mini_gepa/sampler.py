import random
from typing import Any, Iterator, List
from itertools import islice
from pydantic import BaseModel, ConfigDict, PrivateAttr

class InfiniteIdxSampler(BaseModel):
    minibatch_size: int
    trainset_size: int
    seed: int = 0
    epoch: int = 0
    model_config = ConfigDict(extra='allow', validate_assignment=True)
    _iterator: Iterator[int] = PrivateAttr()
    
    def model_post_init(self, __context: Any) -> None:
        assert self.trainset_size > 0
        assert self.minibatch_size > 0
        assert self.seed is not None
        self._iterator = self._stream_indices()

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)

    def _stream_indices(self) -> Iterator[int]:
        """Yields an infinite stream of shuffled dataset indices."""
        while True:
            rng = random.Random(self.seed + self.epoch)
            indices = list(range(self.trainset_size))
            rng.shuffle(indices)
            yield from indices
            self.epoch += 1
    
    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        return next(self._iterator)

    def next_minibatch_indices(self, **kwargs: Any) -> List[int]:
        """Return the next minibatch of indices by consuming from the infinite iterator."""
        return list(islice(self, self.minibatch_size))

if __name__ == "__main__":
    sampler = InfiniteIdxSampler(minibatch_size=10, trainset_size=100, seed=42)
    for i in range(10):
        print(sampler.next_minibatch_indices())

    print('-' * 100)
    for i in range(10):
        print(next(sampler))
