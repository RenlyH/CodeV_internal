# CodeV Evaluation Suite

The `eval/` folder packages a trimmed VLMEvalKit stack for running large-scale multimodal benchmarks with LMDeploy-compatible APIs.  The pipeline expects you to expose your checkpoint through an OpenAI-style HTTP server (LMDeploy, vLLM, or SGLang) and then drive the datasets listed in `vllm_codev.json` via `run.py`.

## Directory Layout
- `run.py`: entry script that loads configs, builds API-backed models, and dispatches inference jobs (image, video, and math benchmarks).
- `vllm_codev.json`: default configuration that wires the `LMDeployAPIWithToolUse` client to multiple datasets (MMMU, MathVista, HRBench, etc.).
- `vlmeval/`: VLMEvalKit modules
  - `api/`: API adapters (LMDeploy, OpenAI, Azure, etc.) plus tool-use wrappers.
  - `vlm/`: local checkpoint runners when you evaluate without HTTP inference.
  - `dataset/`: dataset builders (`ImageMCQDataset`, `MathVerse`, `CharXiv`, …).
  - `inference*.py`: orchestration utilities for single-turn, video, and multi-turn evaluation.
- `requirements.txt`: dependencies for running the evaluator.
- `setup.py`: optional editable install (`pip install -e .`) if you want to import `vlmeval` elsewhere.

## Installation
```bash
cd eval
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
pip install -e .                                    # optional, enables `vlmutil`
```

## Serve the Model (LMDeploy / vLLM / SGLang)
1. Export your model through an OpenAI-compatible REST endpoint.  The default config assumes vLLM:
   ```bash
   vllm serve codev_checkpoint \
       --tensor-parallel-size 4 \
   ```
2. If you prefer LMDeploy or SGLang, start the equivalent server (`lmdeploy serve ...` or `python -m sglang.launch_server ...`) and note the base URL.
3. Update `vllm_codev.json -> model -> codev -> api_base` to match the endpoint (e.g., `http://localhost:8000/v1/chat/completions`).  The vLLM client supports tool-use via `<code>...</code>` blocks, so keep `use_tool: true` if you want Python execution.

## Run Evaluations
1. (Optional) Edit `vllm_codev.json`:
   - Add or remove datasets under `data`.
   - Tweak inference parameters (`temperature`, `max_tokens`, retry limits, tool tokens).
2. Launch:
   ```bash
   cd eval
   python run.py --config vllm_codev.json
   ```
   `run.py` will:
   - Parse the config and instantiate `LMDeployAPIWithToolUse`.
   - Build each dataset listed under `data`.
   - Execute and cache predictions; some datasets will produce intermediate TSV/JSON files.
3. Outputs land under `outputs/` (default inside the repo) with subfolders per dataset/model; adjust by passing `--work-dir` if needed.

## Notes
- The script auto-detects local GPUs and can shard work across processes when `WORLD_SIZE`/`LOCAL_WORLD_SIZE` are set.
- You can override config names directly on the CLI (`python run.py --model codev --data MathVista_MINI`), but configs keep multi-dataset runs reproducible.
- To evaluate local checkpoints instead of HTTP APIs, replace `LMDeployAPIWithToolUse` with one of the classes in `vlmeval.vlm` and make sure dependencies (torch, transformers, etc.) are installed.
