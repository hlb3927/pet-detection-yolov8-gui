from ultralytics import YOLO

def main():
    model = YOLO("../../models/yolov8s.pt")
    model.info(verbose=True)

    project_dir = "../../runs"
    exp_name = "yolov8s_epoch100_img640"

    model.train(
        data="D:\pets_detect\config\pats.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        project="runs",
        mosaic=1.0,             #增加遮挡鲁棒性
        mixup = 0.1,            #增加复杂背景
        hsv_h = 0.015,          #HSV增强，解决背景相似问题
        hsv_s = 0.7,
        hsv_v = 0.4,
        degrees=5,              #增强姿态变化
        translate = 0.1,
        scale = 0.5,
        shear = 0.0,
        conf=0.2,               #降低致信度，增加召回
        name="yolov8s_epoch100_img640",
        exist_ok=True
    )

    print("\nTraining finished.")
    print(f"Check outputs in: {project_dir}/{exp_name}")

if __name__ == "__main__":
    main()