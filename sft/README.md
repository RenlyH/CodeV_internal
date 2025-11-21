# CodeV SFT Toolkit

This folder hosts the lightweight SFT stack, so we can fine-tune models.

## Layout
- `swift/`: the stripped Swift training library (dataset loaders, model registry, schedulers, UIs).
- `scripts/sft_stage1_H200.sh`: minimal launcher example; adjust paths/GPUs to match your environment before running.
- `requirements/` & `requirements.txt`: dependency pins; install with `pip install -r requirements.txt` and pull extras from `requirements/*.txt` if you need optional features.
- `setup.py`: package metadata so the module can be installed in editable mode (`pip install -e .`).

## Quick Start
1. `cd sft && pip install -r requirements.txt`.
2. Prepare SFT JSONL data, which follow this JSON format example (full dataset includes similar structures):
```json
{
  "image": ["/path/to/original.jpg", "/path/to/processed.jpg"],
  "question": "<image>\nBased on the top-right graph, describe the behavior of P(z) as z approaches zero. Options:\n...",
  "response": "<think>Detailed reasoning and executable code...</think><answer>B</answer>"
}
```

### 3.2 Configure Training Paths

Set these variables in your training script or environment:

* `DATASET`: Path to your training dataset
* `SAVE_PATH`: Directory to save the trained model
* `Model`: Path to your model
3. Customize and run `scripts/sft_stage1_H200.sh`, or import `swift` directly to build a bespoke trainer.
