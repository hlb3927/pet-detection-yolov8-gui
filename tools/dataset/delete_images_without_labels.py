from pathlib import Path

#遍历图片文件夹与标签文件夹，删除没有对应标签文件夹的图片
def delete_images_without_label(images_path,labels_path,dry_run=False):
    total_images=0                                          #统计图片总数
    deleted_images=0                                        #统计删除总数

    img_dir=Path(images_path)
    label_dir=Path(labels_path)

    valid_suffixes={".jpg",".jpeg",".png",".bmp",".webp"}   #限制图片处理后缀

    for img in img_dir.iterdir():                           #Path.iterdir()遍历目录下所有文件
        if not img.is_file():
            continue
        if img.suffix.lower() not in valid_suffixes:        #检查后缀是预定后缀，跳过非预定文件
            continue
        img_name=img.stem                                   #Path.stem提取无后缀文件名
        total_images+=1
        label=label_dir/(img_name+".txt")

        if not label.exists():                              #Path.exist()文件存在
            if not dry_run:                                 #输入False或True绝地给是否确定删除
                img.unlink()                                #Path.unlinlk()删除文件
            deleted_images+=1
            print(f"已删除:{img.name}")
    print(f"共删除：{deleted_images}")
    print(f"共有图片:{total_images}")

def main():
    images_path= r"/data/yolo_old\train\images"
    labels_path= r"/data/yolo_old\train\labels"
    delete_images_without_label(images_path,labels_path)

if __name__=="__main__":
    main()


