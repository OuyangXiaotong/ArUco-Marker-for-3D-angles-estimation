import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
global R
# mtx = np.array([
#         [2946.48,       0, 1980.53],
#         [      0, 2945.41, 1129.25],
#         [      0,       0,       1],
#         ])
# #我的手机拍棋盘的时候图片大小是 4000 x 2250
# #ip摄像头拍视频的时候设置的是 1920 x 1080，长宽比是一样的，
# #ip摄像头设置分辨率的时候注意一下
#
#
# dist = np.array( [0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892] )


# 相机纠正参数

# dist=np.array(([[-0.51328742,  0.33232725 , 0.01683581 ,-0.00078608, -0.1159959]]))
#
# mtx=np.array([[464.73554153, 0.00000000e+00 ,323.989155],
#  [  0.,         476.72971528 ,210.92028],
#  [  0.,           0.,           1.        ]])
#import caculateangle

dist = np.array(([[ 0.16285088, -0.22159294,  0.00103373,  0.0010233,   0.0743687 ]]))
newcameramtx = np.array([[1.78165613e+03,0.00000000e+00,1.13784789e+03],
                         [0.00000000e+00,8.97238672e+03,3.19104003e+02],
                         [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
mtx = np.array([[375.59232499,0,326.65323449],
                [0,       374.89607017,183.20463121],
                [0,         0,          1]])

# dist = np.array(([[9.85504698e-02, 6.81623047e-01, -2.86386103e-04, 4.10300557e-03,-4.18069589e+00]]))
# newcameramtx = np.array([[745.28405762,  0.,         664.14438283],
#                         [  0.,         750.26898193, 366.35958279],
#                         [  0.,           0.,           1.        ]])
# mtx = np.array([[745.86678548,  0.,        664.66365553],
#                 [  0.,        745.95955597,366.21618009],
#                 [  0.,          0.,          1.        ]] )

# dist = np.array(([[2.25291918e-01,-6.47644437e-01,1.34044426e-03,3.47655364e-04,7.59953114e-01]]))
# newcameramtx = np.array([[880.41925049,  0.,        639.94703487],
#                         [  0.,        890.00909424,368.23340517],
#                         [  0.,          0.,          1.        ]])
# mtx = np.array([[742.67520754,  0.,        654.74668031],
#                 [  0.,        742.62949976,371.91979465],
#                 [  0.,          0.,          1.        ]] )

cap = cv2.VideoCapture(1)
# cv2.VideoCapture.get(3)
# #在视频流的帧的宽度
# cv2.VideoCapture.get(4)
cap.set(3, 720)  # width=1920
cap.set(4, 1280)
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

# num = 0
while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]
    # 读取摄像头画面
    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    frame = dst1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    '''
    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    axislist = []
    # for i in range(0, 4):
    #     for j in range(0, 2):
    #         axislist.append(rejectedImgPoints[0][0][i][j])
    #         # print(rejectedImgPoints[0][0][i][j])
    #         # print(axislist)
    # print(axislist)


    #    如果找不打id
    if ids is not None:

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        #aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
        #aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
        ###### DRAW ID #####
        cv2.putText(frame, "Id: " + str(ids[0][0]), (0, 64), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(rvec[i, :, :], tvec[i, :, :])
        cv2.putText(frame,str(rvec[0][0]),(0, 96), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,str(tvec[0][0]),(0, 128), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(frame,str(rejectedImgPoints),(0,150),font,0.5 ,(0,255,0),2,cv2.LINE_AA)
        #cv2.putText(frame, str(caculateangle.R), (0, 150), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        deg = rvec[0][0][2] / math.pi * 180
        # deg=rvec[0][0][2]/math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec, R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:  # 偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        #偏航，俯仰，滚动换成角度
        Rx = x * 180.0 / math.pi
        if Rx > 0:
            Rx = 180 - Rx
        elif Rx == 0:
            Rx = 0
        else:
            Rx = -180 - Rx
        #Rx = math.atan(R[2, 1] / R[2, 2]) / math.pi * 180


        Ry = y * 180.0 / math.pi
        Rz = z * 180.0 / math.pi
        angle = np.array([Rx, Ry, Rz])
        cv2.putText(frame, str(R[0]), (0, 150), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(R[1]), (0, 165), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(R[2]), (0, 180), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(angle), (0, 195), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(frame, 'deg_z:' + str(rx) + str('deg'), (0, 140), font, 0.5, (0, 255, 0), 1,
        #           cv2.LINE_AA)
        #distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100  # 单位是米
        # distance = (tvec[0][0][2]) * 100  # 单位是米

        # 显示距离
        #cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 210), font, 0.5, (255, 255, 255), 1,cv2.LINE_AA)






    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示结果框架
    cv2.imshow("frame", frame)

    key = cv2.waitKey(500)

    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        #        num = num + 1
        #        filename = "frames_%s.jpg" % num  # 保存一张图像

        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
        print(R)
        time.sleep(1)
        # axislist = []
        # for i in range(0, 4):
        #     for j in range(0, 2):
        #         axislist.append(rejectedImgPoints[0][0][i][j])
