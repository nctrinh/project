#!/bin/bash
# Host-side parity test: C++ port vs the Python pipeline in src1/ (no FPGA, no torch).
#   pip install numpy opencv-python-headless numba pyyaml
#   ./build.sh mock && tests/parity/run_parity.sh [work_dir]
# Checks: (1) DPU head decode + NMS on random int8 tensors,
#         (2) full motion-branch pipeline (motion core, filtering, merge, FSM,
#             tentative tracks, Kalman, metrics) with replayed detections,
#             for the repo config and a variant with Kalman enabled,
#         (3) pipelined (multi-threaded) run == sequential run.
set -e
FPGA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
REPO_DIR="$( cd "$FPGA_DIR/.." && pwd )"
WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK"
PY=${PYTHON:-python3}
P="$FPGA_DIR/tests/parity"
BIN="$FPGA_DIR/antiuav_fpga_mock"
status=0

echo "== work dir: $WORK"
echo "== (1) decode parity"
# (a) all candidates before NMS (iou=1.0 keeps everything), many int8 score ties -> order-free compare
# (b) tie-free scores -> exact NMS output order
for seed in 0 1 2 3; do
    for kind in prenms nms; do
        D="$WORK/decode_${kind}_$seed"
        if [ $kind == prenms ]; then EXTRA="--iou 1.0 --max-det 100000"; CMP=decode-unordered; else EXTRA="--distinct"; CMP=decode; fi
        $PY "$P/make_decode_case.py" --out "$D" --seed $seed $EXTRA
        $PY "$P/run_python.py" decode --meta "$D/meta.json" --out "$D/py.json"
        "$FPGA_DIR/tests/test_decode" "$D/meta.json" > "$D/cpp.json"
        echo -n "  $kind seed $seed: "
        $PY "$P/compare.py" $CMP "$D/py.json" "$D/cpp.json" || status=1
    done
done

echo "== (2) pipeline parity"
$PY "$P/make_synthetic.py" --out "$WORK/data" --frames 120
SEQ="$WORK/data/synthetic_seq"
DET="$WORK/data/detections_synthetic_seq.json"
cp "$REPO_DIR/configs1/fpga_constraints.yaml" "$WORK/constraints_repo.yaml"
sed 's/^  enabled: false/  enabled: true/' "$REPO_DIR/configs1/fpga_constraints.yaml" > "$WORK/constraints_kalman.yaml"
grep -A1 "^kalman:" "$WORK/constraints_kalman.yaml" | grep -q "enabled: true" || { echo "kalman variant not created"; exit 1; }
for cfg in repo kalman; do
    echo "-- config: $cfg"
    $PY "$P/run_python.py" pipeline --seq-dir "$SEQ" --detections "$DET" \
        --constraints "$WORK/constraints_$cfg.yaml" --out "$WORK/py_$cfg.json"
    "$BIN" --seq-dir "$SEQ" --det-replay "$DET" --constraints "$WORK/constraints_$cfg.yaml" \
        --device "$FPGA_DIR/config/device.yaml" --output-dir "$WORK/cpp_$cfg" --dump-frames > "$WORK/cpp_$cfg.log"
    $PY "$P/compare.py" pipeline "$WORK/py_$cfg.json" "$WORK/cpp_$cfg/sequences/synthetic_seq" || status=1
done

echo "== (3) pipelined vs sequential"
"$BIN" --seq-dir "$SEQ" --det-replay "$DET" --constraints "$WORK/constraints_repo.yaml" \
    --device "$FPGA_DIR/config/device.yaml" --output-dir "$WORK/cpp_seq" --dump-frames --sequential --no-video > /dev/null
if cmp -s "$WORK/cpp_repo/sequences/synthetic_seq/frames_dump.json" "$WORK/cpp_seq/sequences/synthetic_seq/frames_dump.json"; then
    echo "[threading] pipelined and sequential outputs are byte-identical"
else
    echo "[threading] MISMATCH between pipelined and sequential runs"; status=1
fi

[ $status -eq 0 ] && echo "== PARITY OK" || echo "== PARITY FAILED"
exit $status
