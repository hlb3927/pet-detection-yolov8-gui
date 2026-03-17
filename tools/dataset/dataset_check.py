# from pathlib import Path
#
#
# def final_check(img_train,img_val,img_test,label_train,label_val,label_test):
#     img_train = Path(img_train)
#     img_val = Path(img_val)
#     img_test = Path(img_test)
#     label_train = Path(label_train)
#     label_val = Path(label_val)
#     label_test = Path(label_test)
#
#     miss_label=0
#     wrong_label=0
#     overflow_label=0
#     bbox_error = 0
#
#     valid_suffixes={".jpg",".jpeg",".png",".bmp",".webp"}
#     CLASSES = ["horse", "rabbit", "hamster", "guinea pig", "lizard", "bird", "turtle", "dog", "cat", "fish"]
#
#     if not img_train.exists():
#         print(f"训练图像路径不存在")
#         return
#     if not img_val.exists():
#         print(f"验证图像路径不存在")
#         return
#     if not img_test.exists():
#         print(f"测试图像路径不存在")
#         return
#     if not label_train.exists():
#         print(f"训练标签路径不存在")
#         return
#     if not label_val.exists():
#         print(f"验证标签路径不存在")
#         return
#     if not label_test.exists():
#         print(f"测试标签路径不存在")
#         return
#
#     for train_img in img_train.iterdir():
#         if not train_img.is_file():
#             continue
#         if train_img.suffix.lower() not in valid_suffixes:
#             continue
#         train_img_name=train_img.stem
#         train_label=label_train/(train_img_name+".txt")
#         if not train_label.exists():
#             print(f"图片{train_img.name}标签不存在")
#             miss_label+=1
#             continue
#         with open(train_label,'r',encoding='utf-8') as f:
#             lines=f.readlines()
#             for line in lines:
#                 line=line.strip()
#                 parts=line.split()
#                 if len(parts)!=5:
#                     print(f"标签{train_label.name}格式错误")
#                     wrong_label+=1
#                 class_id=int(parts[0])
#                 if class_id<0 or class_id>=len(CLASSES):
#                     print(f"图片{train_img.name}标签溢出")
#                     overflow_label+=1
#                 x_center = float(parts[1])
#                 y_center = float(parts[2])
#                 box_w = float(parts[3])
#                 box_h = float(parts[4])
#                 if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
#                     print(f"标签{train_label.name}box{class_id}错误")
#                     bbox_error+=1
#
#     for val_img in img_val.iterdir():
#         if not val_img.is_file():
#             continue
#         if val_img.suffix.lower() not in valid_suffixes:
#             continue
#         val_img_name=val_img.stem
#         val_label=label_val/(val_img_name+".txt")
#         if not val_label.exists():
#             print(f"图片{val_img.name}标签不存在")
#             miss_label+=1
#             continue
#         with open(val_label,'r',encoding='utf-8') as f:
#             lines=f.readlines()
#             for line in lines:
#                 line=line.strip()
#                 parts=line.split()
#                 if len(parts)!=5:
#                     print(f"标签{val_label.name}格式错误")
#                     wrong_label+=1
#                 class_id=int(parts[0])
#                 if class_id<0 or class_id>=len(CLASSES):
#                     print(f"图片{val_img.name}标签溢出")
#                     overflow_label+=1
#                 x_center = float(parts[1])
#                 y_center = float(parts[2])
#                 box_w = float(parts[3])
#                 box_h = float(parts[4])
#                 if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
#                     print(f"标签{val_label.name}box{class_id}错误")
#                     bbox_error+=1
#
#     for test_img in img_test.iterdir():
#         if not test_img.is_file():
#             continue
#         if test_img.suffix.lower() not in valid_suffixes:
#             continue
#         test_img_name=test_img.stem
#         test_label=label_test/(test_img_name+".txt")
#         if not test_label.exists():
#             print(f"图片{test_img.name}标签不存在")
#             miss_label+=1
#             continue
#         with open(test_label,'r',encoding='utf-8') as f:
#             lines=f.readlines()
#             for line in lines:
#                 line=line.strip()
#                 parts=line.split()
#                 if len(parts)!=5:
#                     print(f"标签{test_label.name}格式错误")
#                     wrong_label+=1
#                     continue
#                 class_id=int(parts[0])
#                 if class_id<0 or class_id>=len(CLASSES):
#                     print(f"图片{test_img.name}标签溢出")
#                     overflow_label+=1
#                 x_center = float(parts[1])
#                 y_center = float(parts[2])
#                 box_w = float(parts[3])
#                 box_h = float(parts[4])
#                 if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
#                     print(f"标签{test_label.name}box{class_id}错误")
#                     bbox_error+=1
#     print(f"标签文件缺失{miss_label}张")
#     print(f"标签格式错误{wrong_label}张")
#     print(f"标签溢出{overflow_label}张")
#     print(f"标签box错误{bbox_error}个")
#
# def main():
#     img_train=r"D:\pets_detect\data\yolo\images\train"
#     img_val=r"D:\pets_detect\data\yolo\images\val"
#     img_test=r"D:\pets_detect\data\yolo\images\test"
#     label_train =r"D:\pets_detect\data\yolo\labels\train"
#     label_val = r"D:\pets_detect\data\yolo\labels\val"
#     label_test = r"D:\pets_detect\data\yolo\labels\test"
#     final_check(img_train,img_val,img_test,label_train,label_val,label_test)
#
# if __name__=="__main__":
#     main()

from pathlib import Path


def check_one_split(img_dir, label_dir, split_name, classes, valid_suffixes):
    """
    检查单个数据集划分（train / val / test）
    返回该 split 的统计结果
    """
    miss_label = 0
    wrong_label = 0
    overflow_label = 0
    bbox_error = 0

    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in valid_suffixes:
            continue

        label_path = label_dir / (img_path.stem + ".txt")

        # 1. 检查标签是否存在
        if not label_path.exists():
            print(f"[{split_name}] 图片 {img_path.name} 对应标签不存在")
            miss_label += 1
            continue

        # 2. 读取并检查标签内容
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines, start=1):
            line = line.strip()

            # 空行直接跳过
            if not line:
                continue

            parts = line.split()

            # 3. 检查字段数
            if len(parts) != 5:
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行格式错误：{line}")
                wrong_label += 1
                continue

            # 4. 检查 class_id
            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 class_id 不是整数：{parts[0]}")
                wrong_label += 1
                continue

            if class_id < 0 or class_id >= len(classes):
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行类别越界：{class_id}")
                overflow_label += 1
                continue

            # 5. 检查 bbox 数值
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
            except ValueError:
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 bbox 不是数字：{line}")
                wrong_label += 1
                continue

            # 6. 检查 bbox 合法性
            if not (0 <= x_center <= 1):
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 x_center 超范围：{x_center}")
                bbox_error += 1

            if not (0 <= y_center <= 1):
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 y_center 超范围：{y_center}")
                bbox_error += 1

            if not (0 < box_w <= 1):
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 box_w 非法：{box_w}")
                bbox_error += 1

            if not (0 < box_h <= 1):
                print(f"[{split_name}] 标签 {label_path.name} 第 {line_idx} 行 box_h 非法：{box_h}")
                bbox_error += 1

    return miss_label, wrong_label, overflow_label, bbox_error


def final_check(img_train, img_val, img_test, label_train, label_val, label_test):
    img_train = Path(img_train)
    img_val = Path(img_val)
    img_test = Path(img_test)
    label_train = Path(label_train)
    label_val = Path(label_val)
    label_test = Path(label_test)

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = ["horse", "rabbit", "hamster", "guinea pig", "lizard", "bird", "turtle", "dog", "cat", "fish"]

    # 1. 先检查路径是否存在
    if not img_train.exists():
        print("训练图像路径不存在")
        return
    if not img_val.exists():
        print("验证图像路径不存在")
        return
    if not img_test.exists():
        print("测试图像路径不存在")
        return
    if not label_train.exists():
        print("训练标签路径不存在")
        return
    if not label_val.exists():
        print("验证标签路径不存在")
        return
    if not label_test.exists():
        print("测试标签路径不存在")
        return

    # 2. 分别检查 train / val / test
    train_result = check_one_split(img_train, label_train, "train", classes, valid_suffixes)
    val_result = check_one_split(img_val, label_val, "val", classes, valid_suffixes)
    test_result = check_one_split(img_test, label_test, "test", classes, valid_suffixes)

    # 3. 汇总统计
    total_miss_label = train_result[0] + val_result[0] + test_result[0]
    total_wrong_label = train_result[1] + val_result[1] + test_result[1]
    total_overflow_label = train_result[2] + val_result[2] + test_result[2]
    total_bbox_error = train_result[3] + val_result[3] + test_result[3]

    print("\n========== 最终检查结果 ==========")
    print(f"标签文件缺失: {total_miss_label} 个")
    print(f"标签格式错误: {total_wrong_label} 处")
    print(f"类别 id 越界: {total_overflow_label} 处")
    print(f"bbox 数值错误: {total_bbox_error} 处")


def main():
    img_train = r"D:\pets_detect\data\yolo\images\train"
    img_val = r"D:\pets_detect\data\yolo\images\val"
    img_test = r"D:\pets_detect\data\yolo\images\test"
    label_train = r"D:\pets_detect\data\yolo\labels\train"
    label_val = r"D:\pets_detect\data\yolo\labels\val"
    label_test = r"D:\pets_detect\data\yolo\labels\test"

    final_check(img_train, img_val, img_test, label_train, label_val, label_test)


if __name__ == "__main__":
    main()