import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a COCO-pretrained YOLOv3n model
    model = YOLO('../model/yolov3_answer.pt')

    # Display model information (optional)
    model.info()
    # Train the model
    results = model.train(data=r'D:\pycharm\F46060\data\trainData\trainData_yolo\data.yaml',
                          epochs=200,
                          imgsz=640,
                          batch=8,
                          )