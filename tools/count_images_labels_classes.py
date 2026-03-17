from pathlib import Path
import cv2



def count_images_labels_classes(images_path, label_path):

    image_dir=Path(images_path)
    label_dir=Path(label_path)

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    CLASSES = ["horse", "rabbit","hamster", "guinea pig", "lizard", "bird", "turtle", "dog", "cat","fish"]
    CLASSES_ID = {c: i for i, c in enumerate(CLASSES)}
    images_count=0
    labels_dir={i:0 for i in range(len(CLASSES))}       #创建标签类别字典用于存放每个类别数量

    if not image_dir.exists():
        print("图片路径不存在")
        return
    if not label_dir.exists():
        print("标签路径不存在")
        return

    for img in image_dir.iterdir():
        if not img.is_file():                           #判断是否是文件
            continue
        if img.suffix.lower() not in valid_suffixes:    #判断后缀是否是所需
            continue
        image = cv2.imread(str(img))                    #cv2.imread(filename[,flags])用于读取图像文件，作为numpy数组返回
        if image is None:
            print(f"无法读取图片")
            continue
        images_count+=1

        image_name=img.stem
        label=label_dir/(image_name+".txt")

        if not label.exists():
            print(f"标签不存在")
            continue

        with open (label,"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                line=line.strip()
                parts=line.split()
                if len(parts) !=5:
                    print(f"标注格式错误{label.name}")
                    continue
                class_id=int(parts[0])
                if class_id<0 or class_id>=len(CLASSES):
                    print(f"标注类别溢出")
                    continue
                labels_dir[class_id]+=1
    for key in range(len(CLASSES)):
        print(f"{CLASSES[key]} 出现了 {labels_dir[key]} 次")
    print(f"共{images_count}张图片")

def main():
    images_path = r"D:\pets_detect\data\yolo\images\test"
    labels_path = r"D:\pets_detect\data\yolo\labels\test"
    count_images_labels_classes(images_path, labels_path)

if __name__=="__main__":
    main()




