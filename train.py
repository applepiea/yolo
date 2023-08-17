import torch
from ultralytics import YOLO
import ultralytics

# print(torch.cuda.is_available())
# print('=========')
# print(ultralytics.checks())

if __name__=='__main__':
	model = YOLO('yolov8n.pt')
	model.train(data='C:/Users/user/PycharmProjects/yolo/Fish-44/data.yaml', imgsz=640, batch=4, epochs=30,device=0) # gpu 쓸 시 넣어줌 , gpu가 두개면 device=[0,1] 등
# yaml 파일 경로, 메모리 에러가 뜨면 batch size를 줄여주기 