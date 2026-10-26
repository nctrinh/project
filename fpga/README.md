# fpga/ - Anti-UAV SOT (motion branch) on a Vitis AI DPU board, C++

C++ port of `scripts/eval_tracker_motion.py` + `src1/antiuav_sot` (same algorithm,
same config keys). YOLOv8 runs on the DPU (`model/yolov8_det.xmodel`); decode/NMS,
motion branch, fusion, Simple Tracker FSM, tentative tracks, Kalman and metrics run
on the ARM CPU. Structure follows `trackerv2-master@9ad897e9251/` (main / DPU / tracker, build.sh, trace_cfg.json).

## 1. Configure the device
* `cp ../compiled/yolov8_det.xmodel model/` (compiled with `scripts/compile.sh` for **your** board's arch.json).
* Edit `config/device.yaml` - every board-specific value is there, marked `TODO(board)`
  (board name, DPU arch/fingerprint, number of DPU cores -> `dpu.num_runners`, threads, video codec).
* `config/fpga_constraints.yaml` is a copy of `configs1/fpga_constraints.yaml` (algorithm
  parameters, validated exactly like `FpgaConstraints.validate()`); keep both in sync or pass `--constraints`.

## 2. Build
* On the board (Vitis AI 3.5 image: VART, XIR, glog, OpenCV): `./build.sh`
* Cross: `source <sdk>/environment-setup-cortexa72-cortexa53-xilinx-linux && ./build.sh`
* Host without DPU (development / parity test): `./build.sh mock`

## 3. Run
Data layout = the one used by `eval_tracker_motion.py` (`data_manifest.json`, `<root>/<seq>/*.jpg + IR_label.json`).
```
./run_sequence.sh /path/data_manifest.json <seq_name> results [--root-dir /path/Anti-UAV/Train_Track1_2/train]
./run_sequence.sh --seq-dir /path/<seq_name> results
./run_eval.sh /path/data_manifest.json val results [--root-dir ...] [--no-video] [--max-sequences N]
```
Outputs: `results/sequences/<seq>/<seq>.mp4` (overlay), `<seq>.json` (`{"res": [[x,y,w,h]|[], ...]}`),
`eval_<seq>.json` (metrics, curves, latency per stage) and `results/eval_results_<split|seq>.json`
(aggregated, same keys as the Python `eval_results_<split>.json`).
"step latency" = compute time per frame (comparable to the Python `latency.mean_ms`);
"throughput" = end-to-end FPS with I/O, video and frame overlap.

Profiling: `vaitrace -c trace_cfg.json` (edit the command line inside first).

## 4. Parity test (host, no FPGA)
`./build.sh mock && tests/parity/run_parity.sh` (needs `numpy opencv-python-headless<5 numba scipy pyyaml`):
decode/NMS vs `dpu_export.decode_outputs`, full pipeline vs `SotPipeline` on a synthetic
sequence with replayed detections (repo config + Kalman on), pipelined == sequential.
