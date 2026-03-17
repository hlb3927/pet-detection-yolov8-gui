import argparse
from pathlib import Path
from core.predictor import YOLOPredictor
import json


def parse_args():
    parser=argparse.ArgumentParser(description="宠物识别")        #参数解析器，用来理解命令行参数，description是脚本说明
    parser.add_argument("--weights",type=str,required=True,help="权重路径")
    #给解释器添加具体参数，参数名是weights，数据类型为str，required=True必须有该参数
    parser.add_argument("--source", type=str, required=True,help="检测图片路径")
    parser.add_argument("--save_dir", type=str, default="outputs", help="保存路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度")
    parser.add_argument("--json_path",type=str,default="outputs",help=".json文件保存路径")
    args=parser.parse_args()                                     #执行解析命令行参数,将参数存入args
    #可以用args.weights、args.source、args.save_dir、args.conf调用参数
    return args

def check_paths(weights,source,save_dir):
    if not weights.exists():
        raise FileNotFoundError(f"权重路径{weights}不存在")
    if not weights.is_file():
        raise ValueError(f"{weights}非法的权重文件")
    if not source.exists():
        raise FileNotFoundError(f"检测图片路径{source}不存在")
    if source.is_dir():
        print(f"文件夹模式")
    elif source.is_file():
        print(f"单图模式")
    else:
        raise ValueError(f"非法图片路径{source}")
    save_dir.mkdir(parents=True,exist_ok=True)


def collect_images(source):
    valid_suffix={".jpg",".jpeg",".png",".bmp",".webp"}
    image_paths=[]
    if source.is_file():
        image_paths=[source]
    elif source.is_dir():
        for img in source.iterdir():
            if img.suffix.lower() not in valid_suffix:
                continue
            image_paths.append(img)
        #print(source/img.name)
        image_paths=sorted(image_paths)
    if len(image_paths) ==0:
        raise ValueError("未读取到图片")
    return image_paths

def infer_images(predictor, image_paths, save_dir, conf,json_path):
    all_results = []
    total_images = len(image_paths)
    total_detections = 0
    images_with_detections = 0
    for idx,img in enumerate(image_paths,start=1):              #enumerate提供循环计数
        print(f"{idx}/{total_images} 预测图像{img.name}")
        result_info=predictor.predict_image(img,save_dir, conf)
        num_dets=result_info["num_dets"]
        total_detections+=num_dets
        all_results.append(result_info)
        if num_dets>0:
            images_with_detections+=1
        print(f"image:{result_info['image_name']},num_dets:{result_info['num_dets']}")
              # f"save_path:{result_info['save_path']},detections:{result_info['detections']}\n\n")
    json_path.mkdir(parents=True,exist_ok=True)
    json_file = json_path / "predictions.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"JSON结果已保存: {json_file}")
    print(f"预测结束")
    print("Inference Summary：")
    print(f"图片总数量: {total_images}")
    print(f"预测框总数: {total_detections}")
    print(f"含预测框图片数: {images_with_detections}")

def main():
    args=parse_args()
    weights=Path(args.weights)
    source=Path(args.source)
    save_dir=Path(args.save_dir)
    json_path=Path(args.json_path)
    conf = args.conf
    check_paths(weights,source,save_dir)
    # model=load_model(weights)
    predictor=YOLOPredictor(weights)
    image_paths = collect_images(source)
    # first_img=image_paths[0]
    # infer_one_image(model, first_img, save_dir, conf)
    infer_images(predictor,image_paths,save_dir, conf,json_path)
    # print(f"模型加载成功")
    # print("weights:", args.weights)
    # print("source:", args.source)
    # print("save_dir:", args.save_dir)
    # print("conf:", args.conf)
    # print(f"num images: {len(image_paths)}")
    # for img in image_paths:
    #     count+=1
    #     print(f"{count}/{len(image_paths)} {img}")

if __name__=="__main__":
    main()