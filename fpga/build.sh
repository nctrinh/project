#!/bin/bash
# Build antiuav_fpga. Same structure as trackerv2/build.sh:
#   ./build.sh          native build on the board, or cross build when CXX points into a
#                       Vitis AI sysroot (e.g. source environment-setup-cortexa72-cortexa53-xilinx-linux)
#   ./build.sh mock     host build WITHOUT VART (mock DPU, use --det-replay) + tests/test_decode
#
# TODO(board): if the board image is not the Vitis AI 3.5 image the library names
# below may differ (check: ls /usr/lib/libvart-*.so /usr/lib/libxir.so).

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}
MODE=${1:-board}
$CXX --version | head -1

result=0 && pkg-config --list-all | grep opencv4 >/dev/null && result=1
if [ $result -eq 1 ]; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi
OPENCV_LIBS="-lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core"

SRC="src/main.cpp \
     src/common/config.cpp src/common/json.cpp \
     src/detector/yolo_dpu.cpp \
     src/motion/gradient_motion.cpp src/motion/adaptive_size.cpp \
     src/tracker/track_state.cpp src/tracker/simple_tracker_fsm.cpp src/tracker/kalman.cpp \
     src/pipeline/sot_pipeline.cpp src/pipeline/sequence_runner.cpp \
     src/eval/dataset.cpp src/eval/metrics.cpp src/eval/viz.cpp"

# No -ffast-math: keep IEEE float semantics so results match the Python reference.
CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -pthread -I$PWD/src"

set -e
if [ "$MODE" == "mock" ]; then
    $CXX $CXXFLAGS -DDPU_MOCK -o antiuav_fpga_mock $SRC src/dpu/dpu_runner_mock.cpp \
        ${OPENCV_FLAGS} ${OPENCV_LIBS} -lpthread
    $CXX $CXXFLAGS -o tests/test_decode tests/test_decode.cpp src/detector/yolo_dpu.cpp src/common/json.cpp \
        src/common/config.cpp src/dpu/dpu_runner_mock.cpp ${OPENCV_FLAGS} ${OPENCV_LIBS} -lpthread
    echo "built: antiuav_fpga_mock, tests/test_decode"
    exit 0
fi

VART_LIBS="-lvart-runner -lxir -lglog -lunilog"
if [[ "$CXX" == *"sysroot"* ]]; then
    # cross compile (paths inside the Vitis AI sysroot, as in trackerv2)
    $CXX $CXXFLAGS \
        -I=/install/Debug/include -I=/install/Release/include \
        -L=/install/Debug/lib -L=/install/Release/lib \
        -o antiuav_fpga $SRC src/dpu/dpu_runner_vart.cpp \
        ${VART_LIBS} ${OPENCV_FLAGS} ${OPENCV_LIBS} -lpthread
else
    os=`lsb_release -a 2>/dev/null | grep "Distributor ID" | sed 's/^.*:\s*//'`
    os_version=`lsb_release -a 2>/dev/null | grep "Release" | sed 's/^.*:\s*//'`
    arch=`uname -p`
    install_prefix_default=$HOME/.local/${os}.${os_version}.${arch}
    $CXX $CXXFLAGS \
        -I${install_prefix_default}.Debug/include -I${install_prefix_default}.Release/include \
        -L${install_prefix_default}.Debug/lib -L${install_prefix_default}.Release/lib \
        -Wl,-rpath=${install_prefix_default}.Debug/lib -Wl,-rpath=${install_prefix_default}.Release/lib \
        -o antiuav_fpga $SRC src/dpu/dpu_runner_vart.cpp \
        ${VART_LIBS} ${OPENCV_FLAGS} ${OPENCV_LIBS} -lpthread
fi
echo "built: antiuav_fpga"
