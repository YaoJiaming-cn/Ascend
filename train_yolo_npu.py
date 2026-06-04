import os
import sys
sys.warnoptions.append('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch_npu


# Monkey-patch: ultralytics 内部多处硬编码 torch.cuda，在 NPU 上会崩溃
from ultralytics.engine.trainer import BaseTrainer

_orig_clear_memory = BaseTrainer._clear_memory

def _patched_clear_memory(self, fraction=None):
    if self.device.type == 'npu':
        return
    return _orig_clear_memory(self, fraction)

BaseTrainer._clear_memory = _patched_clear_memory

# 现在可以安全导入
from ultralytics import YOLO


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU device: {torch.npu.get_device_name(0)}")

    model = YOLO('yolov8n.pt')

    model.train(
        data='coco128/coco128.yaml',
        epochs=10,
        imgsz=320,
        batch=8,
        device='npu:0',
        workers=4,
        amp=False,
        project='yolov8_npu',
        name='train',
        exist_ok=True,
    )

    metrics = model.val(device='npu:0')
    print(f"\nmAP@50:    {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")

    print("\n训练完成！输出在 runs/detect/yolov8_npu/train/")
    print("  权重:     weights/best.pt")
    print("  训练曲线: results.png")


if __name__ == '__main__':
    main()
