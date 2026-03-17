from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from core.predictor import YOLOPredictor
from pathlib import Path
import sys
from core.result_exporter import export_result_to_json
#先创造控件，再放进布局
#使用QHBoxLayout让按钮并排
#使用QVBoxLayout让界面从上到下
def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(".") / relative_path

class DemoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("宠物识别Demo")
        self.resize(1000,700)
        self.image_path=None
        self.weights_path = resource_path("runs/yolov8s_epoch100_img640/weights/best.pt")
        self.save_dir = Path("demo_outputs")
        self.conf=0.25
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.predictor =None
        self.last_result = None
        #YOLOPredictor(self.weights_path)
#创建控件
        self.title_label = QLabel("Pet Detection Local Demo")

        self.select_btn=QPushButton("选择图片")                 #QPushButton创建按钮
        self.detect_btn=QPushButton("开始检测")
        self.export_btn = QPushButton("导出结果")


        self.image_label=QLabel("待显示图片")                    #QLabel图片显示区域
        self.image_label.setMinimumSize(800,450)                #setMiniumSize空间最小尺寸
        self.image_label.setStyleSheet("border:1px solid gray;")    #
        self.image_label.setAlignment(Qt.AlignCenter)           #图片居中

        self.result_text=QTextEdit()                           #QTextEdit()结果文本区
        self.result_text.setReadOnly(True)                     #setReadOnly只读
        self.result_text.setPlainText("检测结果将在这里显示")

#事件连接
        self.select_btn.clicked.connect(self.select_image)      #clicked按钮被点击事件，connect()把事件连接到某个函数
        self.detect_btn.clicked.connect(self.detect_image)
        self.export_btn.clicked.connect(self.export_result)

#创建布局
        button_layout = QHBoxLayout()                           #QHBoxLayout水平布局
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.export_btn)

        main_layout = QVBoxLayout()                             #QVBoxLayo垂直水平布局
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_text)

#主布局
        self.setLayout(main_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(         #打开文件对话框，返回用户选中文件路径与过滤器信息，通常用_接住
            self,                                           #self父窗口
            "选择一张图片",                                    #标题
            "",                                             #初始目录
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)")     #过滤器，只显示图片
        if file_path:
            self.image_path = Path(file_path)
            self.result_text.setPlainText(f"已选择图片：{self.image_path}")#更新文本框内容
            pixmap=QPixmap(str(self.image_path))                 #创建图片对象
            pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio                          #保持比例
            )
            self.image_label.setPixmap(pixmap)              #将图片对象输入

    # def detect_image(self):
    #     if not self.image_path:
    #         self.result_text.setPlainText("请选择一张图片")
    #         return
    #     result_info = self.predictor.predict_image(
    #         self.image_path,
    #         self.save_dir,
    #         self.conf
    #     )
    #     pixmap = QPixmap(result_info["save_path"])
    #     pixmap = pixmap.scaled(
    #         self.image_label.width(),
    #         self.image_label.height(),
    #         Qt.KeepAspectRatio
    #     )
    #     self.image_label.setPixmap(pixmap)
    #     lines=[]
    #     lines.append(f"图片: {result_info['image_name']}")
    #     lines.append(f"检测框数量: {result_info['num_dets']}")
    #     lines.append(f"推理耗时: {result_info['elapsed_time']} s")
    #     lines.append("")
    #     if result_info["num_dets"] == 0:
    #         lines.append("未检测到目标")
    #     else:
    #         for i,det in enumerate(result_info["detections"],start=1):
    #             lines.append(f"目标 {i}")
    #             lines.append(f"类别: {det['cls_name']}")
    #             lines.append(f"置信度: {det['conf_score']}")
    #             lines.append("")
    #     self.result_text.setPlainText("\n".join(lines))
    def detect_image(self):
        if self.predictor is None:
            self.result_text.setPlainText("正在加载模型，请稍候...")
            self.predictor = YOLOPredictor(self.weights_path)
        if not self.image_path:
            self.result_text.setPlainText("请选择一张图片")
            return

        try:
            result_info = self.predictor.predict_image(
                self.image_path,
                self.save_dir,
                self.conf
            )

            if not result_info.get("success", False):
                QMessageBox.warning(self, "检测失败", "模型未成功返回结果。")
                return

            save_path = result_info.get("save_path", "")
            if not save_path:
                QMessageBox.warning(self, "检测失败", "结果图片保存路径为空。")
                return

            pixmap = QPixmap(save_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "显示失败", f"结果图片加载失败：\n{save_path}")
                return

            pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio
            )
            self.image_label.setPixmap(pixmap)

            lines = []
            lines.append(f"图片: {result_info.get('image_name', '')}")
            lines.append(f"检测框数量: {result_info.get('num_dets', 0)}")
            lines.append(f"推理耗时: {result_info.get('elapsed_time', 'unknown')} s")
            lines.append(f"推理设备: {result_info.get('device', 'unknown')}")
            lines.append("")

            if result_info.get("num_dets", 0) == 0:
                lines.append("未检测到目标")
            else:
                for i, det in enumerate(result_info.get("detections", []), start=1):
                    lines.append(f"目标 {i}")
                    lines.append(f"类别: {det.get('cls_name', 'unknown')}")
                    lines.append(f"置信度: {det.get('conf_score', 'unknown')}")
                    lines.append("")

            self.result_text.setPlainText("\n".join(lines))
            self.last_result = result_info

        except Exception as e:
            QMessageBox.critical(self, "程序异常", str(e))


    def export_result(self):
        if self.last_result is None:
            QMessageBox.information(self, "提示", "请先进行检测")
            return

        try:
            json_path = export_result_to_json(self.last_result)

            QMessageBox.information(
                self,
                "导出成功",
                f"结果已保存：\n{json_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))