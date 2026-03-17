from pathlib import Path
import random
import shutil

def train_val_test(img_dir,label_dir,img_train,img_val,img_test,label_train,label_val,label_test):
    img_dir=Path(img_dir)
    label_dir=Path(label_dir)
    img_train=Path(img_train)
    img_val=Path(img_val)
    img_test=Path(img_test)
    label_train = Path(label_train)
    label_val = Path(label_val)
    label_test = Path(label_test)

    if not img_dir.exists():
        print("源图片目录不存在")
        return

    if not label_dir.exists():
        print("源标签目录不存在")
        return

    valid_suffixes={".jpg",".jpeg",".png",".bmp",".webp"}
    pairs=[]

    img_train.mkdir(parents=True,exist_ok=True)     #parents=True上级目录不存在时一起创建，exist_ok=True已存在不报错
    img_val.mkdir(parents=True, exist_ok=True)
    img_test.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    label_val.mkdir(parents=True, exist_ok=True)
    label_test.mkdir(parents=True, exist_ok=True)

    for img in img_dir.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in valid_suffixes:
            continue
        img_name=img.stem
        label=label_dir/(img_name+".txt")
        if not label.exists():
            continue

        pairs.append((img,label))
    random.seed(42)                                 #预设随机种子
    random.shuffle(pairs)                           #random.shuffle()用于原地打乱序列顺序

    total=len(pairs)
    train_end=int(total*0.8)
    val_end=int(total*0.9)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    for img, label in train_pairs:
        shutil.copy2(img,img_train/img.name)                                  #shutil.copy2(src, dst)src源文件路径，dst目标路径，复制文件内容、权限、最后访问时间和最后修改时间等元数据
                                                    #shutil.copy()只复制文件内容和权限，不保留最后访问/修改时间
                                                    #shutil.copyfile()仅复制文件内容，目标文件必须包含文件名，且不复制权限和元数据。
        shutil.copy2(label,label_train/label.name)

    for img, label in val_pairs:
        shutil.copy2(img, img_val / img.name)
        shutil.copy2(label, label_val / label.name)

    for img, label in test_pairs:
        shutil.copy2(img, img_test / img.name)
        shutil.copy2(label, label_test / label.name)

    print(f"总样本数{len(pairs)}")
    print(f"train样本数{len(train_pairs)}")
    print(f"val样本数{len(val_pairs)}")
    print(f"test样本数{len(test_pairs)}")
    print("划分完成")

def main():
    images_path= r"/data/yolo_old/train/images"
    labels_path= r"/data/yolo_old/train/labels"
    img_train=r"D:\pets_detect\data\yolo\images\train"
    img_val=r"D:\pets_detect\data\yolo\images\val"
    img_test=r"D:\pets_detect\data\yolo\images\test"
    label_train =r"D:\pets_detect\data\yolo\labels\train"
    label_val = r"D:\pets_detect\data\yolo\labels\val"
    label_test = r"D:\pets_detect\data\yolo\labels\test"
    train_val_test(images_path,labels_path,img_train,img_val,img_test,label_train,label_val,label_test)

if __name__=="__main__":
    main()

