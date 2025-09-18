from __future__ import annotations
from typing import Any, Dict, List
from typing_extensions import Annotated
from pydantic import BeforeValidator, PlainSerializer
import random

Candidate = Dict[str, Any] # e.g. {'system_prompt': '...'}
DataExample = Dict[str, Any] # e.g. {'input': '...', 'output': '...'}

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

def _rng_from_input(v):
    if isinstance(v, random.Random):
        return v
    if isinstance(v, (tuple, list)):           # model_validate(model_dump()) case
        r = random.Random()
        r.setstate(_tuplify(v))
        return r
    raise TypeError("Expected random.Random or a getstate() tuple/list/dict")

def _rng_to_state(rng: random.Random):
    # return list for JSON-friendliness (tuples become lists in JSON anyway)
    state = rng.getstate()
    return _listify(state)

# Type alias for a strictly-validated, JSON-serializable RNG field
# We anchor validators/serializer on `Any` to avoid requiring schema for `random.Random`.
# The validator enforces/constructs a `random.Random`, and serialization always emits its state list.
PydanticRng = Annotated[
    Any,
    BeforeValidator(_rng_from_input),
    PlainSerializer(_rng_to_state, return_type=list, when_used="always"),
]
