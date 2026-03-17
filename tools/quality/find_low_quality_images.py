from pathlib import Path
import cv2



def find_low_quality_images(images_path, label_path, min_width, min_height, blur_thresh, dry_run=True):#min_width:最小宽度阈值,min_height:最小高度阈值,blur_thresh:模糊阈值
    low_img_list = []
    blur_list = []
    image_dir=Path(images_path)
    label_dir=Path(label_path)

    if not image_dir.exists():
        print("图片路径不存在")
        return
    if not label_dir.exists():
        print("标签路径不存在")
        return

    valid_suffixes={".jpg",".jpeg",".png",".bmp",".webp"}

    low_res_img=0                                       #低分辨率图数量
    blur_img=0                                          #模糊图数量
    delete_img=0                                        #删除图数量

    for img in image_dir.iterdir():
        if not img.is_file():                           #判断是否是文件
            continue
        if img.suffix.lower() not in valid_suffixes:    #判断后缀是否是所需
            continue
        image = cv2.imread(str(img))                    #cv2.imread(filename[,flags])用于读取图像文件，作为numpy数组返回
        if image is None:
            print(f"图片不存在")
            continue

        image_name=img.stem
        label=label_dir/(image_name+".txt")

        if not label.exists():
            print(f"标签不存在")
            continue

        height,width=image.shape[:2]
        if width<min_width or height<min_height:
            low_res_img+=1
            print(f"低分辨率图像:{img.name}")
            if not dry_run:
                img.unlink()
                if label.exists():
                    label.unlink()
                    print(f"已删除:{img.name}")
                delete_img+=1
            low_img_list.append(img.name)
            continue

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     #转灰度图像，cv2读取图像为BGR格式，需要转为灰度图像
                                                        #cv2.cvtColor(src,code)用于颜色空间转换
        blur_score=cv2.Laplacian(gray,cv2.CV_64F).var() #cv2.Laplacian()计算图像拉普拉斯二阶导数算子，用于突出图像的边缘与细节
                                                        #cv2.CV_64F指定输出图像数据类型为64位浮点数，.var()对计算出的拉普拉斯相应图像求方差
                                                        #blur_score值大：图像边缘细节多，更清晰，反之边缘平滑更模糊
        if blur_score<blur_thresh:
            blur_img+=1
            print(f"{img.name} 的模糊得分: {blur_score}")
            print(f"模糊图像：{img.name}")
            blur_list.append((img.name,blur_score))
            blur_list = sorted(blur_list, key=lambda x: x[1])
            if not dry_run:
                img.unlink()
                label.unlink()
                print(f"已删除:{img.name}")
                delete_img+=1
    with open("../low_res_images.txt", "w", encoding="utf-8") as f:
        for name in low_img_list:
            f.write(name+"\n")
    with open("../blur_images.txt", "w", encoding="utf-8") as f:
        for name, score in blur_list:
            f.write(f"{name}\t{score}\n")


    print(f"低分辨率图片数量: {low_res_img}")
    print(f"模糊图片数量: {blur_img}")
    print(f"实际删除数量: {delete_img}")

def main():
    images_path = r"/data/yolo_old\train\images"
    labels_path = r"/data/yolo_old\train\labels"
    find_low_quality_images(images_path, labels_path,224,224,20)

if __name__=="__main__":
    main()




