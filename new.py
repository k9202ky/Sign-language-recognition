import cv2
import mediapipe as mp
import time
import numpy as np
import math

def calculate_angle(v1, v2, v3):
    v1v2 = [v2[0] - v1[0], v2[1] - v1[1]]
    v2v3 = [v3[0] - v2[0], v3[1] - v2[1]]
    
    length_v1v2 = math.sqrt(v1v2[0] ** 2 + v1v2[1] ** 2)
    length_v2v3 = math.sqrt(v2v3[0] ** 2 + v2v3[1] ** 2)

    dot_product = v1v2[0] * v2v3[0] + v1v2[1] * v2v3[1]

    cos_theta = dot_product / (length_v1v2 * length_v2v3)
    cos_theta = min(max(cos_theta, -1), 1)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def hand_angle(hand_):
    angle_list = []

    joints_to_calculate = [(1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10, 11), (10, 11, 12), (13, 14, 15), (14, 15, 16), (17, 18, 19), (18, 19, 20)]

    for joint in joints_to_calculate:
        angle_ = calculate_angle(hand_[joint[0]], hand_[joint[1]], hand_[joint[2]])
        angle_list.append(angle_)

    return angle_list

cv2.namedWindow('Hand Tracking')
cv2.namedWindow('position')
cv2.namedWindow('angle')

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

handLmsStyle = mpDraw.DrawingSpec(color=(0,100,0), thickness=5) 
handConStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=6) 

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        result = hands.process(imgRGB) 
        
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        white_img_1 = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        white_img_1.fill(255)
        white_img_2 = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        white_img_2.fill(255)  

        hand_angles = []
        
        if result.multi_hand_landmarks:
            for handIdx, handLms in enumerate(result.multi_hand_landmarks): 
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                
                finger_points = []
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
                    
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)       
        
        cv2.imshow('Hand Tracking', img)
        cv2.imshow('position', white_img_1)
        cv2.imshow('angle', white_img_2)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
