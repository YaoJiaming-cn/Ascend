# Ascend
| 配置项 | 值 |
  |--------|-----|
  | 模型 | YOLOv8n（3.15M 参数，8.9 GFLOPs） |
  | 预训练权重 | yolov8n.pt（COCO 预训练） |
  | 数据集 | COCO128（128张图，80类） |
  | 训练轮数 | 10 epochs |
  | 图片尺寸 | 320×320 |
  | Batch size | 8 |
  | 优化器 | AdamW（auto，lr=0.000119） |
  | 混合精度 | 关闭（AMP 不兼容 NPU） |
  | DataLoader workers | 4 |

  硬件环境：

  | 配置项 | 值 |
  |--------|-----|
  | 硬件 | 1 × Ascend 910B4（29.5 GB HBM） |
  | CPU 架构 | aarch64（鲲鹏920） |
  | 云平台 | 华为云 ModelArts（西南-贵阳） |
  | CANN 版本 | 8.5.2 |
