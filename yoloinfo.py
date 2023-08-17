from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')
results = model.predict(source='fish.jpg', show=False, save=True)

for result in results:
		boxes = result.boxes
		print(boxes)

print(boxes.xywh) # 텐서타입, 사용하려면 list 타입으로 바꾸기
print(boxes.cls)

print(results[0].xywh) # 바운딩 박스의 좌표값
# 왜 result가 list로 들어감 ? → 이미지를 한꺼번에 집어넣어서 결과를 얻을 수도 있기 때문

print(results[0].cls) # 바운딩 박스 쳐져있는 class의 number가 나오게 됨