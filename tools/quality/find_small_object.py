from pathlib import Path
import cv2

def find_small_object(images_path,labels_path,class_num,min_box_w,min_box_h,min_area_ratio):
    img_dir=Path(images_path)
    label_dir=Path(labels_path)

    total_images=0
    total_boxes=0
    small_boxes=0
    small_object=[]

    if not img_dir.exists():
        print(f"图片路径不存在")
        return
    if not label_dir.exists():
        print("标签路径不存在")
        return

    valid_suffixes={".jpg",".jpeg",".png",".bmp",".webp"}

    for img in img_dir.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in valid_suffixes:
            continue

        total_images+=1
        image=cv2.imread(str(img))

        if image is None:
            print(f"无法读取图片：{img.name}")
            continue

        img_h,img_w=image.shape[:2]

        image_name=img.stem
        label_name=label_dir/(image_name+".txt")
        if not label_name.exists():
            print(f"标签不存在: {label_name.name}")
            continue

        with open(label_name,'r',encoding="utf-8") as f:
            lines=f.readlines()                                 #.readline读取文件所有行
            for line in lines:
                line=line.strip()                               #.strip去掉文件首尾空白
                if not line:
                    continue

                parts=line.split()                              #.split按空格分割字段,输出的是字符串

                if len(parts)!=5:                               #检查每行是否为5个数字
                    print(f"文件格式错误：{label_name}->{line}")
                    continue

                class_id=int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
                total_boxes+=1

                box_w_px=box_w*img_w
                box_h_px=box_h*img_h
                area_ratio = (box_w_px * box_h_px) / (img_w * img_h)

                if box_w_px<min_box_w or box_h_px<min_box_h or area_ratio<min_area_ratio:
                    print(f"图像：{img.name}含小目标")
                    small_object.append((img.name,class_id,box_w_px,box_h_px,area_ratio))
                    small_boxes+=1
    small_object=sorted(small_object,key=lambda x:x[4])

    with open("../small_objects.txt", "w", encoding="utf-8") as t:
        for img_name, class_id, box_w_px, box_h_px, area_ratio in small_object:
            t.write(f"{img_name}\t{class_id}\t{box_w_px:.2f}\t{box_h_px:.2f}\t{area_ratio:.6f}\n")

    print(f"检查图片数量: {total_images}")
    print(f"检查目标框数量: {total_boxes}")
    print(f"小目标数量: {small_boxes}")

def main():
    images_path= r"/data/yolo_old\train\images"
    labels_path= r"/data/yolo_old\train\labels"
    find_small_object(images_path,labels_path,10,16,16,0.01)

if __name__=="__main__":
    main()





