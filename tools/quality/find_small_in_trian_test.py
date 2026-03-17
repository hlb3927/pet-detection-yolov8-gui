from pathlib import Path
import hashlib                                              #计算文件哈希值

#删除同样图片
def find_same_images(train_images_path, train_label_path, test_images_path, test_labels_path, dry_run=True):
    train_img_dir=Path(train_images_path)
    train_label_dir=Path(train_label_path)
    test_img_dir=Path(test_images_path)
    test_label_dir=Path(test_labels_path)
    delete_same=0

    if not train_img_dir.exists():
        print("训练图片路径不存在")
        return
    if not train_label_dir.exists():
        print("训练标签路径不存在")
        return
    if not test_img_dir.exists():
        print("测试图片路径不存在")
        return
    if not test_label_dir.exists():
        print("测试标签路径不存在")
        return
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_hash_dict={}
    test_hash_ditc={}
    same_img=[]

    for train_img in train_img_dir.iterdir():
        if not train_img.is_file():
            continue
        if train_img.suffix.lower() not in valid_suffixes:
            continue
        with open(train_img,"rb") as f:                           #rb二进制读取
            content=f.read()                                      #读出整个文件
        train_img_hash=hashlib.md5(content).hexdigest()           #hashlib.md5()创建一个MD5哈希对象content，
                                                                   # .hexdigest返回哈希值的十六进制字符串表示
        train_hash_dict[train_img_hash]=train_img

    for test_img in test_img_dir.iterdir():
        if not test_img.is_file():
            continue
        if test_img.suffix.lower() not in valid_suffixes:
            continue
        with open(test_img,'rb') as f:
            content=f.read()
        test_img_hash = hashlib.md5(content).hexdigest()



        if test_img_hash in train_hash_dict:                                 #如果重复则记录
            same_img.append((train_hash_dict[test_img_hash],test_img))      #hash_dict[img_hash]第一次出现的图片，img当前重复图片
            if not dry_run:
                img_name=test_img.stem
                test_img.unlink()
                print(f"已删除：{test_img}")
                label = test_label_dir / (img_name+ ".txt")
                if label.exists():
                    label.unlink()
                    print((f"已删除:{label}"))
                delete_same+=1
            else:
                print(f"[预检查] train中已有: {train_hash_dict[test_img_hash].name}")
                print(f"[预检查] test重复图: {test_img.name}")

    print(f"重复图片数量: {len(same_img)}")
    print(f"共删除:{delete_same}")

def main():
    train_images_path= r"/data/yolo_old\train\images"
    train_labels_path= r"/data/yolo_old\train\labels"
    test_images_path = r"/data/yolo_old\test\images"
    test_labels_path = r"/data/yolo_old\test\labels"

    find_same_images(train_images_path, train_labels_path, test_images_path, test_labels_path)

if __name__=="__main__":
    main()

