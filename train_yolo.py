import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from ultralytics import YOLO


def main():
    os.chdir(r'd:\codeproject\shengteng\workfile')

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")

    model = YOLO('yolov8n.pt')

    # 训练
    model.train(
        data='coco128/coco128.yaml',
        epochs=10,
        imgsz=320,
        batch=8,
        device=0,
        workers=0,
        project='yolov8_local',
        name='train',
        exist_ok=True,
    )

    # 验证
    metrics = model.val()
    print(f"\nmAP@50:    {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")

    print("\n训练完成！输出在 runs/detect/yolov8_local/train/")
    print("  权重:     weights/best.pt")
    print("  训练曲线: results.png")
    print("  混淆矩阵: confusion_matrix_normalized.png")
    print("  验证样本: val_batch0_pred.jpg")


if __name__ == '__main__':
    main()
