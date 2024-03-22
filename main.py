import cv2
import mediapipe as mp
import time
import numpy as np
import math
from gesture_ranges import gesture_ranges  # 導入手勢範圍定義

def calculate_angle(v1, v2, v3):
    #計算第一條向量 v1v2
    v1v2_x = v2[0] - v1[0]
    v1v2_y = v2[1] - v1[1]

    #計算第二條向量 v2v3
    v2v3_x = v3[0] - v2[0]
    v2v3_y = v3[1] - v2[1]

    #計算夾角的分子部分（兩向量的內積）
    dot_product = v1v2_x * v2v3_x + v1v2_y * v2v3_y

    #計算兩向量的長度
    length_v1v2 = math.sqrt(v1v2_x ** 2 + v1v2_y ** 2)
    length_v2v3 = math.sqrt(v2v3_x ** 2 + v2v3_y ** 2)

    #計算夾角的分母部分（兩向量的長度乘積）
    denominator = length_v1v2 * length_v2v3

    #使用 arccos 計算弧度
    if denominator != 0:  # 確保分母不為零
        cos_theta = dot_product / denominator
        cos_theta = min(max(cos_theta, -1), 1)  #確保 cos_theta 在合理範圍內
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
    else:
        angle_deg = 180 #如果分母為零，返回 180 表示無法計算夾角

    return angle_deg  #回傳兩向量之間的夾角（以角度表示）


def hand_angle(hand_):
    angle_list = []

    joints_to_calculate = [
        (1, 2, 3),      #angle 1 ∠2
        (2, 3, 4),      #angle 2 ∠3
        (5, 6, 7),      #angle 3 ∠6
        (6, 7, 8),      #angle 4 ∠7
        (9, 10, 11),    #angle 5 ∠10
        (10, 11, 12),   #angle 6 ∠11
        (13, 14, 15),   #angle 7 ∠14
        (14, 15, 16),   #angle 8 ∠15
        (17, 18, 19),   #angle 9 ∠18
        (18, 19, 20)    #angle 10 ∠19
    ]

    for joint in joints_to_calculate:
        angle_ = calculate_angle(hand_[joint[0]], hand_[joint[1]], hand_[joint[2]])
        angle_list.append(angle_)

    return angle_list

def gesture_predict(hand_ang):
    if not hand_ang:
        return "No Gesture"

    for angles in hand_ang:
        for gesture, angle_ranges in gesture_ranges.items():
            if all(min_angle <= angle <= max_angle for angle, (min_angle, max_angle) in zip(angles, angle_ranges)):
                return gesture

    return "Unknown"


cv2.namedWindow('Hand Tracking')
cv2.namedWindow('position')
cv2.namedWindow('angle')

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands #使用手部模型
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #畫上手部特徵點

#自訂手部繪圖樣式
handLmsStyle = mpDraw.DrawingSpec(color=(0,100,0), thickness=5) 
handConStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=6) 

pTime = 0
cTime = 0

while True:
    ret, img = cap.read() 
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #改成RGB讀取
        result = hands.process(imgRGB) #手部讀取結果
        
        #設定實際座標位置
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        white_img_1 = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        white_img_1.fill(255)
        white_img_2 = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        white_img_2.fill(255)  

        hand_angles = []
        
        if result.multi_hand_landmarks:
            #偵測全部的手 不只一隻手
            for handIdx, handLms in enumerate(result.multi_hand_landmarks): 
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                
                finger_points = []
                #偵測全部手的點座標
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    text_to_display = f"Hand {handIdx + 1}, Point {i}: ({xPos}, {yPos})"  
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 2)
                    finger_points.append((xPos, yPos))
                    y_offset = 30 + i * 20 
                    x_offset = handIdx * int(imgWidth / 2)
                    cv2.putText(white_img_1, text_to_display, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

                if finger_points:
                    finger_angle = hand_angle(finger_points)
                    hand_angles.append(finger_angle)  

            for handIdx, angles in enumerate(hand_angles):
                y_offset = 30
                x_offset = handIdx * int(imgWidth / 2)

                for i in range(len(angles)):
                    angle_text = f"Angle {i + 1}: {angles[i]:.2f}"
                    cv2.putText(white_img_2, angle_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    y_offset += 20
                    
        gesture_text = f"gesture : {gesture_predict(hand_angles)}"  # 輸出手勢是甚麼手勢
        cv2.putText(img, gesture_text, (300, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        #設定和顯示FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)       
        
        cv2.imshow('Hand Tracking', img)
        cv2.imshow('position', white_img_1)
        cv2.imshow('angle', white_img_2)

    #按小寫q離開
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
