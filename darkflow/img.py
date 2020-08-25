import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': './yolov2-tiny.cfg',
    'load': './yolov2-tiny_3000.weights',
    'threshold': 0.1,
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

frame = cv2.imread('test1.jpeg')                   #Here img.jpg is image name Or you can also give the path with extension of the file like jpg,jpeg,png etc
results = tfnet.return_predict(frame)
cnt = 0
arr = []
for color, result in zip(colors, results):
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    arr.append(tl)
    arr.append(br)
    cnt = cnt%2
    label = result['label']
    confidence = result['confidence']
    print(confidence, label, cnt)
    if confidence > 0.3 and label == 'Helmet':
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        cnt = cnt + 1
    if label == 'No_Helmet' and confidence > 0.48:
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(frame, text, arr[cnt], cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 4)
        cnt = cnt + 1
cv2.imwrite('aa.png', frame)
cv2.waitKey(0)