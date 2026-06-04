import torch
import torch_npu
from ultralytics import YOLO
import time

print(f"PyTorch: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device: {torch.npu.get_device_name(0)}")

# 加载本地训练好的权重
model = YOLO('runs/detect/yolov8_npu/train/weights/best.pt')

# 预热
for _ in range(5):
    _ = model('test_bus.jpg', device='npu:0', verbose=False)
torch.npu.synchronize()

# 计时推理
t0 = time.time()
for _ in range(50):
    _ = model('test_bus.jpg', device='npu:0', verbose=False)
torch.npu.synchronize()
npu_time = (time.time() - t0) / 50 * 1000

print(f"\nNPU (Ascend 910B) 单张推理: {npu_time:.1f} ms")
print(f"NPU (Ascend 910B) FPS:       {1000/npu_time:.1f}")

# 保存推理结果可视化
results = model('test_bus.jpg', device='npu:0')
results[0].save('test_bus_result.jpg')
print("\n推理结果已保存到 test_bus_result.jpg")
