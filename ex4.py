import cv2
import torch
from ultralytics import YOLO

# YOLOモデルロード
model = YOLO("yolov8x.pt")

# 画像推論
results = model.predict("barcapic.png", conf=0.1)

# 入力画像
img = results[0].orig_img

# 検出ボックス
boxes = results[0].boxes

# 最大面積とそのボックス用の変数
max_area = 0
max_box = None

# 面積最大のボックスを探す
for box in boxes:
    x1, y1, x2, y2 = box.data[0][0:4]
    area = (x2 - x1) * (y2 - y1)
    if area > max_area:
        max_area = area
        max_box = (int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()))

# 最大のボックスだけ描画
if max_box:
    cv2.rectangle(img, (max_box[0], max_box[1]), (max_box[2], max_box[3]), (0, 0, 255), thickness=3)

# 画像保存
cv2.imwrite("out_max_area.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
