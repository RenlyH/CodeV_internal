# CodeV

This repo aggregates the components we use for CodeV training, including, supervised fine-tuning, reinforcement learning, and evaluation.  It stitches together battle-tested open-source stacks, MS-Swift (SFT), VeRL (RL), and VLMEvalKit (evaluation) with a few utility scripts for anonymization and dependency management.

> **Note on provenance & privacy**  
> Many folders are cloned or forked from upstream public repos (`ms-swift`, `verl`, `VLMEvalKit`).  We scrubbed obvious personal/organizational information, but traces may remain in third-party license headers or dataset descriptions.  CVPR’s double-blind policy motivates this warning; if reviewers notice residual identifiers, please understand they originate from required upstream licenses rather than intentional disclosure.

## Repository Layout
- `sft/` – Supervised fine-tuning stack (Swift) with data loaders, trainer configs, and minimal launch scripts.  See `sft/README.md`.
- `rl/` – VeRL-based reinforcement learning pipeline for tool-augmented vision-language agents.  Includes rollout workers, judge integration, and conversion utilities.  See `rl/README.md`.
- `eval/` – Evaluation toolkit built on VLMEvalKit with LMDeploy/vLLM/SGLang API clients.  See `eval/README.md`.
- `assets/` – Supporting figures/tables for papers or reports.
- `requirements*.txt` – Dependency snapshots (global compatibility, RL-specific, etc.).
- `install.sh`, `migrate.sh`, `cleanup_for_cvpr.sh`, `auto_anonymize.sh` – helper scripts for environment setup and anonymization.

## Getting Started
1. **Set up environments**  
   - SFT: `cd sft && pip install -r requirements.txt` (details inside sub README).  
   - RL: `cd rl && pip install -r requirements.txt` plus extras if you need full agent training.  
   - Eval: `cd eval && pip install -r requirements.txt`.
2. **Prepare checkpoints + datasets** following the guidance in each subdirectory.
3. **Run**  
   - SFT: customize `scripts/sft_stage1_H200.sh` or import `swift` directly.  
   - RL: start judge services/Ray cluster, then launch `examples/agent/codev_7b_*.sh`.  
   - Eval: serve your model through LMDeploy/vLLM/SGLang and execute `python run.py --config vllm_codev.json`.
