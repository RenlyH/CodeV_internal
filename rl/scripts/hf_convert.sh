#!/usr/bin/env bash
set -euo pipefail


BASE=""


usage() {
  echo "Usage: $0 [-b BASE_DIR]" >&2
  exit 1
}

while getopts ":b:h" opt; do
  case ${opt} in
    b) BASE="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
  esac
done
shift $((OPTIND - 1))

################################################################################

for STEP_DIR in "$BASE"/global_step_*; do
  [[ -d "$STEP_DIR" ]] || continue    # skip non‑dirs / nothing found

  LOCAL_DIR="$STEP_DIR/actor"
  HF_PATH="$LOCAL_DIR/huggingface"

  echo "🔄  Merging model in: $STEP_DIR"
  python scripts/model_merger.py \
        --local_dir      "$LOCAL_DIR" \
        --hf_model_path  "$HF_PATH"
done
