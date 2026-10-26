#!/bin/bash
# Run every sequence of a manifest split: per-sequence video/JSON/eval
# + aggregated eval_results_<split>.json.
#   ./run_eval.sh <data_manifest.json> [val|train|all] [output_dir] [extra args...]
# e.g. ./run_eval.sh /data/experiments/data_manifest.json val results --root-dir /data/Anti-UAV/Train_Track1_2/train
#      add --no-video to skip video encoding, --max-sequences N for a quick run.
cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
BIN=${BIN:-./antiuav_fpga}
if [ $# -lt 1 ]; then
    echo "usage: $0 <data_manifest.json> [val|train|all] [output_dir] [extra args]"; exit 1
fi
MANIFEST=$1; SPLIT=${2:-val}; OUT=${3:-results}; shift 3 2>/dev/null || shift $#
exec $BIN --manifest "$MANIFEST" --split "$SPLIT" --output-dir "$OUT" "$@"
