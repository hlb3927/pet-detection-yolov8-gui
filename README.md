# Pet Object Detection Based on YOLOv8

## 1. Project Overview
基于 YOLOv8 的 10 类宠物目标检测项目，覆盖 horse / rabbit / hamster / guinea pig / lizard / bird / turtle / dog / cat / fish。

## 2. Dataset Engineering
- 原始数据泄漏检查
- 基于 MD5 的重复样本检测
- 数据集按 8:1:1 重构
- 标签格式校验
- 类别统计与数据质量分析

## 3. Training Pipeline
- baseline: YOLOv8n
- model comparison: YOLOv8n vs YOLOv8s
- input size experiment: 640 vs 940
- augmentation experiment: YOLOv8s + aug + 100 epochs

## 4. Results
- Best model: YOLOv8s + aug + 100 epochs
- mAP50: 0.900
- mAP50-95: 0.733

## 5. Error Analysis
- 搭建自动 Error Analysis 工具
- 输出 TP/FP/FN 按类别统计
- 自动保存 FN / FP 可视化样本
- 重点分析 fish 类漏检原因

## 6. Project Highlights
- 完整的数据工程能力
- 规范的对照实验设计
- 自动化错误分析工具
- 具备算法与工程结合能力