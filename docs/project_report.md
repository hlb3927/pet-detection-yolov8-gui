# Experiment Log

## Exp-01 baseline_yolov8n_640_e50
- model: yolov8n
- imgsz: 640
- epochs: 50
- result:
  - P: 0.903
  - R: 0.777
  - mAP50: 0.881
  - mAP50-95: 0.690
- observation:
  - fish recall明显偏低
  - hamster/dog存在漏检

## Exp-02 baseline_yolov8s_640_e50
- model: yolov8s
- imgsz: 640
- epochs: 50
- result:
  - P: 0.877
  - R: 0.838
  - mAP50: 0.893
  - mAP50-95: 0.721
- observation:
  - v8s整体优于v8n
  - fish召回提升但仍偏低

## Exp-03 yolov8s_940_e50
- model: yolov8s
- imgsz: 940
- epochs: 50
- result:
  - P: 0.904
  - R: 0.764
  - mAP50: 0.870
  - mAP50-95: 0.667
- observation:
  - 大尺寸输入未带来收益
  - fish性能进一步下降

## Exp-04 yolov8s_aug_epoch100
- model: yolov8s
- imgsz: 640
- epochs: 100
- aug:
  - mosaic=1.0
  - mixup=0.1
  - hsv_h=0.015
  - hsv_s=0.7
  - hsv_v=0.4
- result:
  - P: 0.901
  - R: 0.837
  - mAP50: 0.900
  - mAP50-95: 0.733
- observation:
  - 当前最佳结果
  - fish仍然是核心难类