# -*- coding: utf-8 -*-
"""
YOLOv8 小麦检测训练脚本
正式训练：80 轮 + patience=15 早停（冒烟测试已验证 GPU 与数据集连通）
"""
from pathlib import Path

from ultralytics import YOLO

# 当前脚本所在目录（即项目根目录），保证无论从哪运行都能找到 data.yaml
BASE_DIR = Path(__file__).resolve().parent


def main():
    # 加载轻量级预训练模型 yolov8n.pt（首次运行会自动下载）
    model = YOLO("yolov8n.pt")

    # 开始训练
    # 注意：project 传绝对路径；传相对路径时 ultralytics 会再拼一层默认的 runs/detect 前缀
    model.train(
        data=str(BASE_DIR / "data.yaml"),     # 数据集配置文件（当前目录下的 data.yaml）
        epochs=80,                            # 正式训练轮数
        patience=15,                          # 早停：验证指标连续 15 轮无提升则提前结束
        imgsz=640,                            # 输入图像尺寸
        batch=16,                             # 批次大小（若显存不足可降为 8 或 4）
        device=0,                             # 使用本地 NVIDIA 显卡（cuda:0）加速
        project=str(BASE_DIR / "runs" / "train"),  # 训练输出根目录
        name="wheat_formal",                  # 本次实验名称 -> runs/train/wheat_formal
    )

    # 训练结束后，打印最佳权重 best.pt 的完整保存路径
    # 注意：train() 返回的是指标对象，best.pt 路径需从 trainer 上取
    best_pt = Path(model.trainer.best).resolve()
    print("=" * 60)
    print(f"训练完成！最佳权重 best.pt 保存路径：\n{best_pt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
