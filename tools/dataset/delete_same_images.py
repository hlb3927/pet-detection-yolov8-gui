from pathlib import Path
import hashlib                                              #计算文件哈希值

#删除同样图片
def delete_same_images(images_path,label_path,dry_run=False):
    img_dir=Path(images_path)
    label_dir=Path(label_path)
    delete_same=0
    if not img_dir.exists():
        print("图片路径不存在")
        return
    if not label_dir.exists():
        print("标签路径不存在")
        return
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    hash_dict={}
    same_img=[]
    for img in img_dir.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in valid_suffixes:
            continue
        with open(img,"rb") as f:                           #rb二进制读取
            content=f.read()                                #读出整个文件
        img_hash=hashlib.md5(content).hexdigest()            #hashlib.md5()创建一个MD5哈希对象content，
                                                            # .hexdigest返回哈希值的十六进制字符串表示
        if img_hash in hash_dict:                           #如果重复则记录
            same_img.append((hash_dict[img_hash],img))      #hash_dict[img_hash]第一次出现的图片，img当前重复图片
            if not dry_run:
                img_name=img.stem
                img.unlink()
                print(f"已删除：{img}")
                label = label_dir / (img_name+ ".txt")
                if label.exists():
                    label.unlink()
                    print((f"已删除:{label}"))
                delete_same+=1
        else:
            hash_dict[img_hash]=img
    print(f"唯一图片数量: {len(hash_dict)}")
    print(f"重复图片数量: {len(same_img)}")
    print(f"共删除:{delete_same}")

def main():
    images_path= r"/data/yolo_old\train\images"
    labels_path= r"/data/yolo_old\train\labels"
    delete_same_images(images_path,labels_path)

if __name__=="__main__":
    main()

