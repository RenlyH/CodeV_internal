#!/usr/bin/env bash
set -euo pipefail


BASE="../../exp/verl_checkpoints/agent_vlagent/debug_vstar_chart4_thinklite_micro4_H200_micro4_stopsign"


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

# echo "✅  All merges finished."
echo $BASE
echo ${BASE#/tmp/instance_storage/xinhaih/}

aws s3 sync "$BASE"  "s3://xinhaih-2025-summer/xinhaih/${BASE#/tmp/instance_storage/xinhaih/}" \
  --exclude "*" \
  --include "global_step_*/actor/huggingface/**" \
  --profile greenland \
  # --dryrun
