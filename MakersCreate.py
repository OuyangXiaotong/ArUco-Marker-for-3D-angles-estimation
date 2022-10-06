# import cv2 as cv
# import numpy as np
#
# # 加载用于生成标记的字典
# dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
#
# # Generate the marker
# markerImage = np.zeros((200, 200), dtype=np.uint8)
# markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1);
#
# cv.imwrite("marker33.png", markerImage);

import cv2
import numpy as np
import argparse
import sys
# 生成aruco标记
# 加载预定义的字典
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# 生成标记
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage22 = cv2.aruco.drawMarker(dictionary, 22, 200, markerImage, 1)
cv2.imwrite("marker22.png", markerImage22)
markerImage23 = cv2.aruco.drawMarker(dictionary, 23, 200, markerImage, 1)
cv2.imwrite("marker23.png", markerImage23)
markerImage24 = cv2.aruco.drawMarker(dictionary, 24, 200, markerImage, 1)
cv2.imwrite("marker24.png", markerImage24)
markerImage25 = cv2.aruco.drawMarker(dictionary, 25, 200, markerImage, 1)
cv2.imwrite("marker25.png", markerImage25)

