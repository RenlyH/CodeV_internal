# CodeV RL Stack

This directory packages the TAPO pipeline used to train `CodeV`.  It is a VeRL-based agent trainer with asynchronous rollouts, tool-calling, and reward models tuned for “code with images”. Python sandbox design can be found here: (`rl/verl/workers/agent/envs/python_interpreter/python_interpreter.py`) and reward design here: (`rl/verl/utils/reward_score/vl_agent.py`)


## Directory Layout
- `verl/`: fork of VeRL providing trainers, Ray orchestration, rollout workers, reward services, and vLLM backends.
- `examples/agent/`: end-to-end launch scripts (`codev_7b_H200.sh`, `codev_7b_l40s.sh`) that wire up datasets, tool servers, and logging for 7B-scale runs.
- `scripts/`: helper utilities (`model_merger.py`, `converter_hf_to_mcore.py`, `hf_convert.sh`, etc.) plus `sys_prompt.txt` for system instructions.
- `requirements.txt` / `requirements_rl.txt`: core dependencies vs. heavyweight extras (vision toolchains, evaluation utilities).
- `setup.py`: enables editable installs (`pip install -e .`) so `verl` can be imported inside any experiment harness.

## Environment Setup
```bash
cd rl
python -m venv .venv && source .venv/bin/activate          # optional but recommended
pip install -r requirements.txt                            # base stack
pip install -r requirements_rl.txt                         # extras: ray, wandb, high-res vision deps
pip install -e .                                           # expose verl package
wandb login                                                # if you want dashboard logging
```

## Hardware & Service Prerequisites
- **Training GPUs**: ≥8 GPUs for 7B models; each node with ≥1.2 TB RAM due to high-res images.
- **Judge Model**: Run a vLLM server that hosts an evaluation LLM (e.g., `Qwen/Qwen2.5-VL-32B-Instruct`) to score responses.
- **Ray Cluster**: Bring up a Ray head plus workers covering all compute nodes; trainers use Ray for rollout + logging coordination.
- **Datasets**: Download the multimodal RL dataset bundle (JSONL + image tarballs).  Point `DATA_REPO` / `DATA_CACHE` variables in the example scripts to your storage paths.

### Example: Start the Judge Service
```bash

vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --served-model-name judge \
    --trust-remote-code \
    --disable-log-requests
```

### Example: Prepare Training Nodes
```bash
export LLM_AS_A_JUDGE_BASE=http://ip:18901/v1
export DATA_REPO=/path/to/rl_data
export HF_HOME=/path/to/hf_cache
```

## Launching Training
1. Review `examples/agent/codev_7b_H200.sh` (default 7B config). It sets:
   - foundation/checkpoint paths (`BASE_MODEL_PATH`, `VLLM_MODEL_PATH`)
   - dataset shards and replay buffers
   - reward and judge endpoints (`LLM_AS_A_JUDGE_BASE`)
   - logging backends (Weights & Biases + RL Logging Board)
2. Customize environment variables at the top of the script to match your cluster, dataset locations, and credentials.
3. Run `bash examples/agent/codev_7b_H200.sh`.  The script:
   - boots rollout workers across Ray
   - streams samples to VeRL trainers
   - syncs rewards from the judge API
   - writes checkpoints into `outputs/` (path configurable via `--save-path`)

## Utilities
- `scripts/model_merger.py`: merge partitioned checkpoints into Hugging Face format (supports FSDP + Megatron layouts).
- `scripts/converter_hf_to_mcore.py`: convert HF weights into Megatron-Core format for large inference jobs.
- `scripts/hf_convert.sh`: shell wrapper for bulk conversions.
- `scripts/extract_images_from_parquet.py`: utility for dumping raw images when datasets are stored in parquet shards.
- `scripts/sys_prompt.txt`: canonical system prompt injected into policy rollouts.


## Logging & Monitoring
- **Weights & Biases**: enabled by default; set `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_API_KEY`.
- **RL Logging Board**: served via the trainer; exposes reward curves, tool usage, and per-dataset accuracy snapshots.
- **Stdout/TensorBoard**: Ray jobs stream logs under each node’s `~/ray_results` directory for quick debugging.
