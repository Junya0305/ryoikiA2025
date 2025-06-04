from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルのロード
model = YOLO("yolov8x.pt")
img = cv2.imread("ex5.jpg")

# YOLOで物体検出
results = model(img)
boxes = results[0].boxes

# HSV色空間で青色の範囲を定義
lower = np.array([90, 80, 40])  
upper = np.array([140, 255, 255])

def is_blue_area(hsv_img, x1, y1, x2, y2):
    # 領域を切り抜き
    cropped_img = hsv_img[y1:y2, x1:x2]

    # 青色領域のマスクを作成
    mask = cv2.inRange(cropped_img, lower, upper)

    # 青色ピクセルの割合を計算
    blue_ratio = cv2.countNonZero(mask) / (cropped_img.shape[0] * cropped_img.shape[1])

    # 青色領域と判定
    return blue_ratio > 0.03

# HSV色空間に変換
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 検出された領域を確認し、青色領域なら赤枠で描画
for box in boxes:
    x1, y1, x2, y2 = map(int, box.data[0][:4])  # バウンディングボックス座標を取得
    if is_blue_area(hsv_image, x1, y1, x2, y2):  # 青色領域かを判定
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 結果を保存および表示
cv2.imwrite("out.jpg", img)  
cv2.waitKey(0)
cv2.destroyAllWindows()
