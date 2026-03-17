from ultralytics import YOLO
import cv2
import os
from pathlib import Path

model = YOLO(r"/runs/yolov8s_epoch50_img640/weights/best.pt")

img_dir = Path(r"/data/yolo/images/val")
label_dir = Path(r"/data/yolo/labels/val")

save_dir = Path("01fn_cases")
save_dir.mkdir(parents=True, exist_ok=True)

IOU_THRESH = 0.5

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter/union if union>0 else 0


for img_path in img_dir.glob("*.*"):

    img = cv2.imread(str(img_path))
    h,w = img.shape[:2]

    label_path = label_dir/(img_path.stem+".txt")

    if not label_path.exists():
        continue

    gt_boxes = []

    with open(label_path) as f:
        for line in f:
            cls,x,y,bw,bh = map(float,line.split())

            x1 = (x-bw/2)*w
            y1 = (y-bh/2)*h
            x2 = (x+bw/2)*w
            y2 = (y+bh/2)*h

            gt_boxes.append((int(cls),[x1,y1,x2,y2]))

    results = model(img)[0]

    pred_boxes = []

    if results.boxes is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()

        for b,c in zip(boxes,cls):
            pred_boxes.append((int(c),b))

    missed = []

    for gcls,gbox in gt_boxes:

        matched = False

        for pcls,pbox in pred_boxes:

            if gcls!=pcls:
                continue

            if iou(gbox,pbox)>IOU_THRESH:
                matched=True
                break

        if not matched:
            missed.append((gcls,gbox))

    if missed:

        for cls,box in missed:

            x1,y1,x2,y2 = map(int,box)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imwrite(str(save_dir/img_path.name),img)