#!/bin/bash
# Run ONE sequence: video + predictions JSON + eval of that sequence.
#   ./run_sequence.sh <data_manifest.json> <sequence_name> [output_dir] [extra args...]
#   ./run_sequence.sh --seq-dir <path/to/sequence_dir> [output_dir] [extra args...]
# Extra args are passed to antiuav_fpga, e.g. --root-dir /mnt/data/Anti-UAV/Train_Track1_2/train
cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
BIN=${BIN:-./antiuav_fpga}
if [ "$1" == "--seq-dir" ]; then
    SEQ_DIR=$2; OUT=${3:-results}; shift 3 2>/dev/null || shift $#
    exec $BIN --seq-dir "$SEQ_DIR" --output-dir "$OUT" "$@"
fi
if [ $# -lt 2 ]; then
    echo "usage: $0 <data_manifest.json> <sequence_name> [output_dir] [extra args]"; exit 1
fi
MANIFEST=$1; SEQ=$2; OUT=${3:-results}; shift 3 2>/dev/null || shift $#
exec $BIN --manifest "$MANIFEST" --split all --sequence "$SEQ" --output-dir "$OUT" "$@"
