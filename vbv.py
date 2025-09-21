import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(1)

data = None
color = None
color_ranges = {'green':  ([40, 40, 40], [80, 255, 255]),
                'blue':  ([90, 50, 50], [130, 255, 255])}

def ct(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dtc = []
    lower = np.array(color_ranges[color][0])
    upper = np.array(color_ranges[color][1])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    pixels = cv2.countNonZero(mask)
    if pixels > 1000:
        dtc.append(color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (255, 255, 255),
                          2)
            dtc.append(color)
    if dtc:
        print(f'color: {color}')
        cv2.putText(frame, f"Colors: {' '.join(dtc)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        if data is not None:
            print(data)
    if not ret :
        print("无法读取画面")
        continue

    if data == '1':
        color = 'blue'
        ct(frame)
    elif data == '2':
        color = 'green'
        ct(frame)

    cv2.imshow('Color Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()