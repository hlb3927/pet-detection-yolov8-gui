import time

from ultralytics import YOLO
import cv2


class YOLOPredictor:
    def __init__(self,weights):
        print(f"加载模型...")
        self.model=YOLO(str(weights))
        print(f"模型加载完成")

    def predict_image(self,img_path,save_dir,conf):      #单张预测，主要预测
        save_dir.mkdir(parents=True, exist_ok=True)
        statr_time=time.time()
        detections=[]

        img=cv2.imread(str(img_path))

        if img is None:
            raise ValueError("图片读取失败")

        result=self.model(img,conf=conf)
        result=result[0]

        num_dets=len(result.boxes)

        save_path=save_dir/img_path.name

        print(f"save to:{save_path}")

        for box in result.boxes:
            cls_id=int(box.cls[0])
            conf_score=round(float(box.conf[0]),2)              #round(x,y)将x保留y位小数
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = {
                "xmin": round(x1, 2),
                "ymin": round(y1, 2),
                "xmax": round(x2, 2),
                "ymax": round(y2, 2)
            }

            cls_name = result.names[cls_id]

            det={
                "cls_id":cls_id,
                "cls_name":cls_name,
                "conf_score":conf_score,
                "bbox":bbox,
            }

            detections.append(det)

        vis_img = result.plot()
        cv2.imwrite(str(save_path), vis_img)
        elapsed=time.time()-statr_time

        result_info={
            "success":True,
            "image_name":img_path.name,
            "save_path":str(save_path),
            "num_dets":num_dets,
            "detections":detections
        }

        return result_info