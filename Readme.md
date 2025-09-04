# simple-gepa

A minimal, readable local optimizer with a simple GEPA-style adapter example on AIME.

Quickstart (requires OPENAI_API_KEY):

```bash
# cd <root>
uv venv -p 3.12 --managed-python
uv pip install -e .
cd examples/demo
python aime_demo.py --model gpt-5 --max-metric-calls 200 --minibatch-size 3 --lanes 8 --run-dir runs/aime_minimal --reasoning-effort low
```

- Dataset: AIME (loaded via Hugging Face `datasets`)
- Metric: response contains the exact substring `### <answer>`
- Actor: OpenAI Responses API (async)
- Optimizer: round-robin over adapter-declared components (default: `system_prompt`), acceptance by mean minibatch score

See `simple_gepa/core.py` for the engine and `examples/demo/placeholder.py` for a fully self-contained adapter and CLI.
