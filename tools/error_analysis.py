"""
error_analysis.py

作用：
1. 对目标检测模型在指定数据集（如 val）上的结果进行自动化误差分析
2. 输出每个类别的 TP / FP / FN 统计
3. 输出每张图片级别的错误统计
4. 自动保存 FN / FP / TP 可视化图片
5. 新增：
   - 混淆统计（GT类别 -> 预测成了什么类别）
   - 置信度分布统计
   - 难例榜单（按 FN / FP 排序）
   - 更详细的 JSON / CSV 报告

适用场景：
- 训练完 YOLOv8 后分析验证集错误
- 重点分析某个类别，例如 fish
- 面试 / 项目汇报时展示“自动化 error analysis 工具能力”

运行方式：
python scripts/error_analysis.py

你也可以在其他脚本中导入本文件的 analyze_dataset() 函数进行调用。
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 1. 全局配置
# =========================================================

# 类别名称列表：
# 下标就是类别 id，例如：
# 0 -> horse
# 1 -> rabbit
# ...
CLASSES = [
    "horse",
    "rabbit",
    "hamster",
    "guinea pig",
    "lizard",
    "bird",
    "turtle",
    "dog",
    "cat",
    "fish"
]

# 支持读取的图片后缀
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# 2. 基础工具函数
# =========================================================

def ensure_dir(path: Path):
    """
    作用：
    确保目录存在；如果不存在则自动创建。

    用法：
    ensure_dir(Path("analysis/reports"))

    参数：
    path: pathlib.Path 对象

    返回：
    无返回值
    """
    path.mkdir(parents=True, exist_ok=True)


def xywhn_to_xyxy(box, img_w, img_h):
    """
    作用：
    将 YOLO 标签常见的归一化格式 [x_center, y_center, width, height]
    转换成像素坐标格式 [x1, y1, x2, y2]

    为什么要做这个转换：
    - GT 标签通常是 YOLO txt 格式（归一化）
    - 模型预测框通常是 xyxy 像素坐标
    - 要做 IoU 匹配，必须统一坐标系

    用法：
    xyxy_box = xywhn_to_xyxy([0.5, 0.5, 0.2, 0.3], img_w=640, img_h=480)

    参数：
    box   : [x_center, y_center, width, height]，数值范围通常在 0~1
    img_w : 图片宽度（像素）
    img_h : 图片高度（像素）

    返回：
    [x1, y1, x2, y2]
    """
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]


def box_area(box):
    """
    作用：
    计算一个 xyxy 框的面积

    用法：
    area = box_area([10, 20, 100, 200])

    参数：
    box: [x1, y1, x2, y2]

    返回：
    面积（float）
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def calc_iou(box1, box2):
    """
    作用：
    计算两个目标框的 IoU（Intersection over Union）

    为什么需要这个函数：
    - 误差分析时需要判断某个预测框是否“命中”GT
    - 一般 IoU > 0.5 认为匹配成功

    用法：
    iou = calc_iou(box1, box2)

    参数：
    box1, box2: 两个 xyxy 格式的框

    返回：
    IoU 值，范围 0~1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(box1) + box_area(box2) - inter

    if union <= 0:
        return 0.0
    return inter / union


def get_size_bucket(box, img_w, img_h):
    """
    作用：
    将目标按面积占整张图的比例分成：
    - small
    - medium
    - large

    为什么要做这个：
    - 后面你可以统计 FN 到底主要发生在哪种尺度
    - 比如你这次 fish 的 FN 大多是 small，这个信息非常关键

    用法：
    bucket = get_size_bucket(box, img_w, img_h)

    参数：
    box   : xyxy 框
    img_w : 图像宽度
    img_h : 图像高度

    返回：
    "small" / "medium" / "large"
    """
    area = box_area(box)
    ratio = area / (img_w * img_h + 1e-9)

    if ratio < 0.01:
        return "small"
    elif ratio < 0.09:
        return "medium"
    else:
        return "large"


# =========================================================
# 3. 读取 GT 和预测结果
# =========================================================

def load_gt_labels(label_path: Path, img_w: int, img_h: int):
    """
    作用：
    读取单张图片对应的 GT 标签文件（YOLO txt 格式）

    返回格式：
    [
        {
            "cls": 类别id(int),
            "box": [x1, y1, x2, y2]
        },
        ...
    ]

    为什么单独写成函数：
    - 便于主流程复用
    - 便于后续替换为别的标注格式（如 COCO）

    用法：
    gts = load_gt_labels(label_path, img_w, img_h)

    参数：
    label_path : 标签文件路径
    img_w      : 图片宽
    img_h      : 图片高

    返回：
    gts 列表
    """
    gts = []

    if not label_path.exists():
        return gts

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"[警告] 标签格式错误: {label_path} 第{line_idx}行 -> {line}")
            continue

        try:
            cls_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            print(f"[警告] 标签解析失败: {label_path} 第{line_idx}行 -> {line}")
            continue

        box = xywhn_to_xyxy([x_c, y_c, w, h], img_w, img_h)
        gts.append({
            "cls": cls_id,
            "box": box
        })

    return gts


def load_predictions(model, img, conf_thres=0.25):
    """
    作用：
    调用 YOLO 模型对单张图片进行预测，并把结果整理成统一格式

    为什么要单独封装：
    - 主流程更清晰
    - 后续可以方便切换 conf_thres
    - 也可以把 predict 参数集中在这里改

    用法：
    preds = load_predictions(model, img, conf_thres=0.25)

    参数：
    model      : ultralytics.YOLO 实例
    img        : cv2.imread() 读到的 BGR 图像
    conf_thres : 预测置信度阈值

    返回：
    [
        {
            "cls": 类别id(int),
            "conf": 置信度(float),
            "box": [x1, y1, x2, y2]
        },
        ...
    ]
    """
    preds = []

    result = model.predict(img, conf=conf_thres, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    for box, c, s in zip(xyxy, cls, conf):
        preds.append({
            "cls": int(c),
            "conf": float(s),
            "box": box.tolist()
        })

    return preds


# =========================================================
# 4. 匹配逻辑
# =========================================================

def greedy_match(gts, preds, iou_thres=0.5):
    """
    作用：
    在 GT 和 Pred 之间做“贪心匹配”

    匹配规则：
    - 同类别才能匹配
    - IoU >= iou_thres 才算匹配成功
    - 一个 GT 最多匹配一个 Pred
    - 一个 Pred 最多匹配一个 GT
    - 预测框按置信度从高到低优先匹配

    为什么这么做：
    - 这是检测分析里很常见的近似做法
    - 简单、清晰、可控
    - 足够用于错误分析

    用法：
    matched_pairs, fn_indices, fp_indices = greedy_match(gts, preds, iou_thres=0.5)

    参数：
    gts       : GT 列表
    preds     : 预测结果列表
    iou_thres : IoU 阈值

    返回：
    matched_pairs : [(gt_idx, pred_idx, iou), ...]
    fn_indices    : 未匹配到的 GT 索引列表（即漏检）
    fp_indices    : 未匹配到的 Pred 索引列表（即误检）
    """
    matched_pairs = []                      #存储匹配成功的（gt_id,prep_id）
    used_gt = set()                         #set()创建一个集合，无序不重复的元素序列
    used_pred = set()

    pred_order = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)
    #将预测结果按置信度从高到低排列，range(len(preds))生成一个整数序列
    #key=lambda i: preds[i]["conf"]指定排序的依据。对于每个索引 i，取出对应预测框的置信度 preds[i]["conf"]，用这个值来比较大小。
    #reverse=True：降序排序，即置信度高的排在前面。sorted(...)：返回一个新的列表，包含按置信度排序后的索引。

    for pred_idx in pred_order:
        pred = preds[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt:
                continue
            if gt["cls"] != pred["cls"]:
                continue

            iou = calc_iou(gt["box"], pred["box"])
            if iou >= iou_thres and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            used_gt.add(best_gt_idx)
            used_pred.add(pred_idx)
            matched_pairs.append((best_gt_idx, pred_idx, best_iou))

    fn_indices = [i for i in range(len(gts)) if i not in used_gt]
    fp_indices = [i for i in range(len(preds)) if i not in used_pred]

    return matched_pairs, fn_indices, fp_indices


def find_best_cross_class_match(gt, preds, used_pred=None):
    """
    作用：
    当某个 GT 没有被“同类别正确匹配”时，尝试寻找它最像哪个“错误类别预测框”
    用于统计类别混淆。

    举例：
    GT 是 fish，但模型没有把它识别成 fish，
    却有一个 hamster 的框和它 IoU 很高，
    那么这可能是 fish -> hamster 的混淆。

    用法：
    best_pred_idx, best_iou = find_best_cross_class_match(gt, preds, used_pred=None)

    参数：
    gt        : 单个 GT 字典
    preds     : 所有预测框
    used_pred : 可选，某些场景下可以传入已用预测框集合

    返回：
    (best_pred_idx, best_iou)
    如果没找到，则返回 (-1, 0.0)
    """
    best_iou = 0.0
    best_pred_idx = -1

    for pred_idx, pred in enumerate(preds):
        if used_pred is not None and pred_idx in used_pred:
            continue

        iou = calc_iou(gt["box"], pred["box"])
        if iou > best_iou:
            best_iou = iou
            best_pred_idx = pred_idx

    return best_pred_idx, best_iou


# =========================================================
# 5. 可视化函数
# =========================================================

def draw_case(img, gts, preds, fn_indices, fp_indices, matched_pairs):
    """
    作用：
    把单张图的匹配情况画出来，便于人工查看

    颜色约定：
    - TP: 绿色
    - FN: 红色
    - FP: 黄色

    用法：
    vis = draw_case(img, gts, preds, fn_indices, fp_indices, matched_pairs)

    参数：
    img           : 原图
    gts           : GT 列表
    preds         : 预测列表
    fn_indices    : 漏检 GT 索引
    fp_indices    : 误检 Pred 索引
    matched_pairs : 成功匹配对

    返回：
    带可视化框的图像
    """
    vis = img.copy()

    # 画 TP
    for gt_idx, pred_idx, iou in matched_pairs:
        gt = gts[gt_idx]
        x1, y1, x2, y2 = map(int, gt["box"])
        cls_name = CLASSES[gt["cls"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"TP {cls_name} IoU={iou:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # 画 FN
    for idx in fn_indices:
        gt = gts[idx]
        x1, y1, x2, y2 = map(int, gt["box"])
        cls_name = CLASSES[gt["cls"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis, f"FN {cls_name}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    # 画 FP
    for idx in fp_indices:
        pred = preds[idx]
        x1, y1, x2, y2 = map(int, pred["box"])
        cls_name = CLASSES[pred["cls"]]
        conf = pred["conf"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis, f"FP {cls_name} {conf:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

    return vis


# =========================================================
# 6. 主分析函数
# =========================================================

def analyze_dataset(
    model_path,
    image_dir,
    label_dir,
    save_root,
    conf_thres=0.25,
    iou_thres=0.5,
    focus_class=None,
    confusion_iou_thres=0.1
):
    """
    作用：
    对整个数据集（通常是 val）做自动误差分析。
    这是本脚本最核心的主函数。

    你平时调用时，最常用的就是这个函数。

    基本用法：
    analyze_dataset(
        model_path="runs/detect/xxx/weights/best.pt",
        image_dir="data/yolo/images/val",
        label_dir="data/yolo/labels/val",
        save_root="analysis/val_error_analysis"
    )

    参数说明：
    model_path         : 模型权重路径
    image_dir          : 图片目录
    label_dir          : 标签目录
    save_root          : 分析结果输出目录
    conf_thres         : 推理置信度阈值
    iou_thres          : TP/FN/FP 匹配时使用的 IoU 阈值
    focus_class        : 重点分析的类别名，如 "fish"，若为 None 表示不聚焦单类
    confusion_iou_thres: 做跨类混淆统计时的最低 IoU 阈值

    输出内容：
    - fn_cases/       : 含 FN 的图
    - fp_cases/       : 含 FP 的图
    - matched_cases/  : 全对的图
    - reports/        : JSON / CSV 报告
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    save_root = Path(save_root)

    fn_dir = save_root / "fn_cases"
    fp_dir = save_root / "fp_cases"
    matched_dir = save_root / "matched_cases"
    report_dir = save_root / "reports"

    ensure_dir(fn_dir)
    ensure_dir(fp_dir)
    ensure_dir(matched_dir)
    ensure_dir(report_dir)

    model = YOLO(model_path)

    # 每类 TP/FP/FN
    class_tp = Counter()                            #counter()用于统计元素的字典子类
    class_fp = Counter()
    class_fn = Counter()

    # 错误尺度分布
    fn_size_counter = defaultdict(Counter)
    fp_size_counter = defaultdict(Counter)

    # 置信度统计：记录每个类别 TP/FP 的预测置信度
    tp_conf_scores = defaultdict(list)
    fp_conf_scores = defaultdict(list)

    # 类别混淆统计：
    # confusion_counter["fish"]["hamster"] = 3
    # 表示 GT fish 被更像 hamster 的框错误覆盖了 3 次
    confusion_counter = defaultdict(Counter)

    # 单图报告
    image_reports = []

    image_paths = [p for p in image_dir.iterdir()                       #遍历所有图片
                   if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    image_paths = sorted(image_paths)

    print(f"共发现 {len(image_paths)} 张图片，开始分析...")

    focus_class_id = None                                               #focus_class 的预处理
    if focus_class is not None:
        if focus_class not in CLASSES:
            raise ValueError(f"focus_class={focus_class} 不在类别列表中")
        focus_class_id = CLASSES.index(focus_class)

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] 分析 {img_path.name}")

        img = cv2.imread(str(img_path))                                 #遍历单张图片，做核心分析
        if img is None:
            print(f"[警告] 无法读取图片: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"

        # 读取 GT 和预测
        gts = load_gt_labels(label_path, img_w, img_h)
        preds = load_predictions(model, img, conf_thres=conf_thres)

        # 做同类匹配，得到 TP / FN / FP
        matched_pairs, fn_indices, fp_indices = greedy_match(gts, preds, iou_thres=iou_thres)

        matched_gt_set = set()
        matched_pred_set = set()

        # 统计 TP
        for gt_idx, pred_idx, _ in matched_pairs:
            matched_gt_set.add(gt_idx)
            matched_pred_set.add(pred_idx)

            gt_cls = gts[gt_idx]["cls"]
            pred_conf = preds[pred_idx]["conf"]

            class_tp[gt_cls] += 1
            tp_conf_scores[gt_cls].append(pred_conf)

        # 统计 FN
        for gt_idx in fn_indices:
            gt_cls = gts[gt_idx]["cls"]
            class_fn[gt_cls] += 1

            size_bucket = get_size_bucket(gts[gt_idx]["box"], img_w, img_h)
            fn_size_counter[gt_cls][size_bucket] += 1

            # 跨类混淆分析：
            # 看这个 GT 有没有和其他类别预测框有一定 IoU
            best_pred_idx, best_iou = find_best_cross_class_match(gts[gt_idx], preds)
            if best_pred_idx != -1 and best_iou >= confusion_iou_thres:
                pred_cls = preds[best_pred_idx]["cls"]
                if pred_cls != gt_cls:
                    confusion_counter[gt_cls][pred_cls] += 1

        # 统计 FP
        for pred_idx in fp_indices:
            pred_cls = preds[pred_idx]["cls"]
            pred_conf = preds[pred_idx]["conf"]

            class_fp[pred_cls] += 1
            fp_conf_scores[pred_cls].append(pred_conf)

            size_bucket = get_size_bucket(preds[pred_idx]["box"], img_w, img_h)
            fp_size_counter[pred_cls][size_bucket] += 1

        # 生成可视化图
        vis = draw_case(img, gts, preds, fn_indices, fp_indices, matched_pairs)

        # 是否包含 focus_class 的 FN
        has_focus_fn = False
        if focus_class_id is not None:
            has_focus_fn = any(gts[i]["cls"] == focus_class_id for i in fn_indices)

        # 保存图片
        if fn_indices:
            if focus_class is None or has_focus_fn:
                cv2.imwrite(str(fn_dir / img_path.name), vis)

        if fp_indices:
            cv2.imwrite(str(fp_dir / img_path.name), vis)

        if matched_pairs and not fn_indices and not fp_indices:
            cv2.imwrite(str(matched_dir / img_path.name), vis)

        # 统计单图中的 focus_class FN 数
        focus_fn_count = 0
        if focus_class_id is not None:
            focus_fn_count = sum(1 for i in fn_indices if gts[i]["cls"] == focus_class_id)

        image_reports.append({
            "image_name": img_path.name,
            "num_gt": len(gts),
            "num_pred": len(preds),
            "num_tp": len(matched_pairs),
            "num_fn": len(fn_indices),
            "num_fp": len(fp_indices),
            "focus_fn": has_focus_fn,
            "focus_fn_count": focus_fn_count
        })

    # =====================================================
    # 汇总 JSON
    # =====================================================
    summary = {
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "focus_class": focus_class,
        "per_class": {},
        "confusion_analysis": {},
        "hard_examples": {}
    }

    for cls_id, cls_name in enumerate(CLASSES):
        tp = class_tp[cls_id]
        fp = class_fp[cls_id]
        fn = class_fn[cls_id]

        precision_like = tp / (tp + fp + 1e-9)
        recall_like = tp / (tp + fn + 1e-9)

        summary["per_class"][cls_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_like": round(precision_like, 4),
            "recall_like": round(recall_like, 4),
            "tp_conf_mean": round(mean(tp_conf_scores[cls_id]), 4) if tp_conf_scores[cls_id] else None,
            "fp_conf_mean": round(mean(fp_conf_scores[cls_id]), 4) if fp_conf_scores[cls_id] else None,
            "fn_size_distribution": dict(fn_size_counter[cls_id]),
            "fp_size_distribution": dict(fp_size_counter[cls_id])
        }

    # 类别混淆分析结果
    for gt_cls_id, pred_counter in confusion_counter.items():
        gt_name = CLASSES[gt_cls_id]
        summary["confusion_analysis"][gt_name] = {
            CLASSES[pred_cls_id]: count
            for pred_cls_id, count in pred_counter.items()
        }

    # 难例榜单：按 FN / FP 排序
    sorted_by_fn = sorted(image_reports, key=lambda x: x["num_fn"], reverse=True)
    sorted_by_fp = sorted(image_reports, key=lambda x: x["num_fp"], reverse=True)
    sorted_by_focus_fn = sorted(image_reports, key=lambda x: x["focus_fn_count"], reverse=True)

    summary["hard_examples"]["top_fn_images"] = sorted_by_fn[:20]
    summary["hard_examples"]["top_fp_images"] = sorted_by_fp[:20]
    summary["hard_examples"]["top_focus_fn_images"] = sorted_by_focus_fn[:20]

    # =====================================================
    # 保存 JSON
    # =====================================================
    json_path = report_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # =====================================================
    # 保存 per_class_summary.csv
    # =====================================================
    csv_path = report_dir / "per_class_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "tp", "fp", "fn", "precision_like", "recall_like",
            "tp_conf_mean", "fp_conf_mean",
            "fn_small", "fn_medium", "fn_large",
            "fp_small", "fp_medium", "fp_large"
        ])

        for cls_name in CLASSES:
            item = summary["per_class"][cls_name]
            writer.writerow([
                cls_name,
                item["tp"],
                item["fp"],
                item["fn"],
                item["precision_like"],
                item["recall_like"],
                item["tp_conf_mean"],
                item["fp_conf_mean"],
                item["fn_size_distribution"].get("small", 0),
                item["fn_size_distribution"].get("medium", 0),
                item["fn_size_distribution"].get("large", 0),
                item["fp_size_distribution"].get("small", 0),
                item["fp_size_distribution"].get("medium", 0),
                item["fp_size_distribution"].get("large", 0),
            ])

    # =====================================================
    # 保存 image_level_report.csv
    # =====================================================
    image_csv_path = report_dir / "image_level_report.csv"
    with open(image_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "num_gt", "num_pred", "num_tp", "num_fn", "num_fp",
            "focus_fn", "focus_fn_count"
        ])
        for row in image_reports:
            writer.writerow([
                row["image_name"],
                row["num_gt"],
                row["num_pred"],
                row["num_tp"],
                row["num_fn"],
                row["num_fp"],
                row["focus_fn"],
                row["focus_fn_count"]
            ])

    # =====================================================
    # 保存 confusion_summary.csv
    # =====================================================
    confusion_csv_path = report_dir / "confusion_summary.csv"
    with open(confusion_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_class", "pred_class", "count"])
        for gt_name, pred_dict in summary["confusion_analysis"].items():
            for pred_name, count in pred_dict.items():
                writer.writerow([gt_name, pred_name, count])

    print("\n分析完成。")
    print(f"JSON报告: {json_path}")
    print(f"按类CSV报告: {csv_path}")
    print(f"单图CSV报告: {image_csv_path}")
    print(f"混淆CSV报告: {confusion_csv_path}")


# =========================================================
# 7. 主程序入口
# =========================================================

if __name__ == "__main__":
    """
    这里是直接运行本文件时的默认入口。

    你平时只需要改下面这几个参数：
    1. model_path  : 你的 best.pt 路径
    2. image_dir   : 要分析的图片目录
    3. label_dir   : 对应标签目录
    4. save_root   : 输出目录
    5. focus_class : 想重点盯哪个类，比如 fish

    运行命令：
    python scripts/error_analysis.py
    """
    analyze_dataset(
        model_path="runs/yolov8s_epoch100_img640/weights/best.pt",
        image_dir="data/yolo/images/val",
        label_dir="data/yolo/labels/val",
        save_root="analysis/val_error_analysis_v2",
        conf_thres=0.25,
        iou_thres=0.5,
        focus_class="fish",
        confusion_iou_thres=0.1
    )
