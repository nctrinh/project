vai_c_xir \
  -x quantize_result/YOLOv8DPU_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled \
  -n yolov8_det

# ZCU102	/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
# ZCU104	/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
# KV260	/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json