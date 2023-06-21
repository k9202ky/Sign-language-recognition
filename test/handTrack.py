import cv2
import mediapipe as mp
import time

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
        
        #print(result.multi_hand_landmarks)
        
        #設定實際座標位置
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        if result.multi_hand_landmarks:
            #偵測全部的手 不只一隻手
            for handLms in result.multi_hand_landmarks: 
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                #break #只偵測一隻手
                
                #偵測全部手的點座標
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 2) #顯示點的編號
                    print(i, xPos, yPos)
                    
        #設定和顯示FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)       
        
        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'): #按小寫q離開
        break
