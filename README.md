# 🐾 YOLOv8 宠物目标检测系统（GUI + 分析 + 部署）

基于 YOLOv8 构建的完整视觉算法工程项目，覆盖：

👉 数据处理 → 模型训练 → 错误分析 → GUI工具 → 本地部署

---

## 🚀 项目亮点（Highlights）

- 自建 **10 类宠物检测数据集（1236 张图像）**
- 完成数据清洗（去重 / 缺失标注 / 低质量 / 小目标分析）
- 按 **8:1:1 划分训练 / 验证 / 测试集**
- 多组实验对比（YOLOv8n / YOLOv8s / 不同输入尺寸）
- 最佳模型性能：
  - **mAP50：0.900**
  - **mAP50-95：0.733**
- 构建 **错误分析工具（FN检测）**
- 开发 **PyQt5 GUI 本地推理工具**
- 使用 PyInstaller 实现 **EXE 部署**

---

## 🧠 项目背景

本项目旨在模拟真实视觉算法工程流程，从零构建一个目标检测系统，并解决实际问题：

- 数据质量问题
- 类别不均衡问题
- 小目标检测问题
- 模型误检 / 漏检问题

---

## 📊 数据集说明

- 类别数：10 类
- 总图片数：1236
- 划分比例：8:1:1

类别包括：
horse / rabbit / hamster / guinea pig / lizard / bird / turtle / dog / cat / fish

---

### 数据处理流程：

- 删除重复图片（MD5）
- 删除无标注图片
- 检查类别越界
- 检测低质量图片
- 小目标统计分析

---

## 🏗️ 项目结构

pets_detect/
├── core/ # 推理模块
├── gui/ # GUI界面
├── analysis/ # 错误分析与数据分析
├── tools/ # 数据处理与训练脚本
├── config/ # 数据配置
├── models/ # 模型权重（本地）
├── runs/ # 训练结果
├── outputs/ # 推理输出
├── app/ # 程序入口
├── requirements.txt
└── README.md

---

## 🤖 模型训练

使用 Ultralytics YOLOv8：

### Baseline：
model = yolov8n
imgsz = 640
epochs = 50

### 实验对比：

- YOLOv8n vs YOLOv8s
- 输入尺寸：640 / 940
- 训练轮数：50 / 100

### 最优结果：
YOLOv8s + 数据增强 + 100 epochs
mAP50: 0.900
mAP50-95: 0.733

---

## 🖥️ GUI 推理工具

基于 PyQt5 实现桌面应用：

### 功能：

- 图片选择
- 目标检测
- 检测结果可视化
- 检测信息展示
- 推理耗时统计

---

## ⚙️ 环境安装

```bash
pip install -r requirements.txt
- 运行gui
python -m app.demo_app
-推理脚本
python tools/infer/infer.py

