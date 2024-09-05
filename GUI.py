# 加载qt5
import glob
import os.path

import cv2
import numpy as np
import multiprocessing
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QMessageBox, QFileDialog, QTableWidgetItem
from PyQt5.QtGui import QIcon, QFont, QPixmap, QStandardItemModel
from PyQt5.uic import loadUi
from UI.main import Ui_Form
from ultralytics import YOLO
from collections import defaultdict

class MyApplication(QWidget, Ui_Form):
    file_name = None
    imgList = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loadUi(r'UI/main.ui', self)

        # self.model_car = YOLO('model/yolov3_根据用户数据集训练的模型.pt')
        self.model_car = YOLO('model/yolov3-tinyu.pt')

        self.choose = 0     # 存放当前检测的图片处于所有图片的第几张。

        self.pushButton_9.clicked.connect(self.DetectCap)
        self.pushButton.clicked.connect(self.DetectVideo)

    def DetectCap(self):
        self.label_14.setText("当前选中摄像头: 0")
        video = cv2.VideoCapture(0)
        # Store the track history
        track_history = defaultdict(lambda: [])
        # 逐帧获得图片
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # 把label设置设置选择的图片
            # self.label.setPixmap(QPixmap(ImgPath))
            originImg = frame.copy()

            results = self.model_car.track(originImg,
                                           verbose=False,
                                           persist=True,
                                           classes=[2], )
            # 转化成列表
            Text = ''
            if results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    Text += f'汽车：{track_id} ,坐标为：x={int(x)}，y={int(y)},w={int(w)},h={int(h)}\n'
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(128, 255, 230), thickness=5)

                # Display the annotated frame
                cv2.imwrite(r'temp.jpg', annotated_frame)
                cv2.imwrite(r'temp2.jpg', originImg)
                self.label.setPixmap(QPixmap(r'temp2.jpg'))
                self.label_16.setPixmap(QPixmap(r'temp.jpg'))
                self.label_2.setText(Text)
            else:
                cv2.imwrite(r'temp.jpg', originImg)
                cv2.imwrite(r'temp2.jpg', originImg)
                self.label.setPixmap(QPixmap(r'temp2.jpg'))
                self.label_16.setPixmap(QPixmap(r'temp.jpg'))
            cv2.waitKey(1)

    def DetectVideo(self):
        # 弹出一个选择mp4文件的窗口
        file_name = QFileDialog.getOpenFileName(self, '选择视频文件', '', "Video Files (*.mp4)")
        self.label_14.setText(file_name[0])
        video = cv2.VideoCapture(file_name[0])
        # Store the track history
        track_history = defaultdict(lambda: [])
        # 逐帧获得图片
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # 把label设置设置选择的图片
            # self.label.setPixmap(QPixmap(ImgPath))
            originImg = frame.copy()
            frame = np.zeros_like(originImg)

            results = self.model_car.track(originImg,
                                           verbose=False,
                                           persist=True,
                                           classes=[2],)
            # 转化成列表
            Text = ''
            if results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box

                    Text += f'汽车：{track_id} ,坐标为：x={int(x)}，y={int(y)},w={int(w)},h={int(h)}\n'
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(128, 255, 230), thickness=5)

                # Display the annotated frame
                cv2.imwrite(r'temp.jpg', annotated_frame)
                cv2.imwrite(r'temp2.jpg', originImg)
                self.label.setPixmap(QPixmap(r'temp2.jpg'))
                self.label_16.setPixmap(QPixmap(r'temp.jpg'))
                self.label_2.setText(Text)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.imwrite(r'temp.jpg', originImg)
                cv2.imwrite(r'temp2.jpg', originImg)
                self.label.setPixmap(QPixmap(r'temp2.jpg'))
                self.label_16.setPixmap(QPixmap(r'temp.jpg'))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication([])
    QToolTip.setFont(QFont('SansSerif', 10))
    app.setStyle('Fusion')
    window = MyApplication()
    window.show()

    app.exec_()
