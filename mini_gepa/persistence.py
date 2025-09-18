from typing import Dict, Any
import json
import os
from pathlib import Path

def _find_largest_index(run_dir: str) -> int:
    run_dir = Path(run_dir)
    largest_index = -1
    for file in run_dir.glob("ckpt_*.json"):
        # Extract the number from ckpt_{number}.json
        name = file.stem  # removes .json extension
        name = name.replace("ckpt_", "")
        index = int(name)
        largest_index = max(largest_index, index)
    return largest_index

def save_state(run_dir: str, state) -> None:
    print(f'saving state to {run_dir}')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    largest_index = _find_largest_index(run_dir)
    # Create new checkpoint file with incremented index
    new_index = largest_index + 1
    ckpt_path = str(run_dir / f"ckpt_{new_index}.json")
    data = state.model_dump()
    with open(ckpt_path, "w") as f:
        f.write(json.dumps(data, indent=2))
    print(f'saved state to {ckpt_path}')

def load_state(run_dir: str):
    from mini_gepa.core import OptimizationState
    largest_index = _find_largest_index(run_dir)
    if largest_index == -1:
        return None
    ckpt_path = Path(run_dir) / f"ckpt_{largest_index}.json"
    dict_data = json.loads(ckpt_path.read_text())
    return OptimizationState.model_validate(dict_data)