from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")
image_path = "data/images/test.jpg"

results = model(image_path)
img = cv2.imread(image_path)

persons = []
motorcycles = []
helmets = []

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))
        elif label == "helmet":
            helmets.append((x1, y1, x2, y2))

def overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

for m in motorcycles:
    for p in persons:
        if overlaps(p, m):
            px1, py1, px2, py2 = p
            head_area = (px1, py1, px2, py1 + int((py2 - py1) * 0.35))

            helmet_on = False
            for h in helmets:
                if overlaps(h, head_area):
                    helmet_on = True

            color = (0, 255, 0) if helmet_on else (0, 0, 255)
            text = "HELMET ON" if helmet_on else "NO HELMET"

            cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
            cv2.putText(img, text, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(img, (m[0], m[1]), (m[2], m[3]), (255, 0, 0), 2)

os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/result.jpg", img)

print("Detection completed. Check outputs/result.jpg")
