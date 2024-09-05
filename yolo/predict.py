import cv2
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv3n model
model = YOLO('yolov3-tinyu.pt')
videoPath = r"D:\pycharm\F46060\data\video\92b229d89da17c3748d0f97bba9730c2.mp4"