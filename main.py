import cv2 as cv
import handtraking_module as htm
import math
import time
import numpy as np


#################################################################
reslution  = (1280,720)
pTime =0

FPS_PT = (10, 700)
MSG_PT = (900, 700)
GIT_PT = (5,120)
PER_PT = (0,0)


C_BLACK =(0,0,0)
C_RED =  (0,0,255)
C_BLUE=  (255,0,0)
C_GREEN =(0,255,0)
DRAW_COLOR = C_RED

MENU = cv.imread(r"banner\menu.png")
RED = cv.imread(r"banner\Red.png")
GREEN = cv.imread(r"banner\Green.png")
BLUE = cv.imread(r"banner\Blue.png")
ERASER = cv.imread(r"banner\Eraser.png")
Bannner = {"MENU":MENU, "RED":RED,"GREEN":GREEN,"BLUE":BLUE,"ERASER":ERASER }
Banner_size= MENU.shape[0:2]
BANNER_NAME="MENU"
Banner_size= Bannner[BANNER_NAME].shape[0:2]


BRUSH_THICKNESS= 15

imgCanvas = np.zeros((720,1280,3),np.uint8)
####################################################################

cap = cv.VideoCapture(1)
cap.set(3,reslution[0])
cap.set(4,reslution[1])

detctor = htm.handDetector(maxHands=1,detectionCon=0.7)

while True:
    suc , img = cap.read()
    img = cv.flip(img , 1)
   
    img = detctor.findHands(img)
    landmark = detctor.findPosition(img,draw=False)
    # print(len(landmark))
    msg="HAND NOT DETECTED"
    if len(landmark) >=10:        
        MID_FING = landmark[8][1:]
        INDEX_FING = landmark[12][1:]
        
        # MID_PT= (abs((MID_FING[0]-INDEX_FING[0])//2),abs(MID_FING[1]-INDEX_FING[1])//2)
        # print(MID_PT) 
        lenght = math.hypot(MID_FING[0]-INDEX_FING[0],MID_FING[1]-INDEX_FING[1])
        # print(lenght)
        if lenght < 60:
            cv.rectangle(img,(MID_FING[0]-10,MID_FING[1]-30), (INDEX_FING[0]+10, INDEX_FING[1]+30), DRAW_COLOR,cv.FILLED )
            msg= "SELECT"
        else:
            cv.circle(img,MID_FING,20, DRAW_COLOR,cv.FILLED )
            msg="DRAW"
        # print(MID_FING)
        if msg =="SELECT":
            PER_PT = (0,0)
            # print(MID_FING)
            if MID_FING[1]<=130:
                # ERASER
                if 1030<MID_FING[0]<1130:
                    BANNER_NAME="ERASER"
                    DRAW_COLOR= C_BLACK
                    BRUSH_THICKNESS= 50
                # RED
                elif 810<MID_FING[0]<910:
                    BANNER_NAME="RED"
                    DRAW_COLOR=C_RED
                    BRUSH_THICKNESS= 15
                # BLUE
                elif 600<MID_FING[0]<700:
                    BANNER_NAME="BLUE"
                    DRAW_COLOR=C_BLUE
                    BRUSH_THICKNESS= 15
                # GREEN
                elif 400<MID_FING[0]<500:
                    BANNER_NAME= "GREEN"
                    DRAW_COLOR=C_GREEN
                    BRUSH_THICKNESS= 15
                else:    
                    BANNER_NAME="MENU"
        elif msg=="DRAW":
            if PER_PT==(0,0):
                PER_PT= MID_FING 
            cv.line(img,PER_PT,MID_FING,DRAW_COLOR,BRUSH_THICKNESS)
            cv.line(imgCanvas,PER_PT,MID_FING,DRAW_COLOR,BRUSH_THICKNESS)
            PER_PT= MID_FING



    # print(BANNER_NAME)
    Banner_size= Bannner[BANNER_NAME].shape[0:2]
    # print(Banner_size)

    img[0:Banner_size[0], 0:Banner_size[1]] = Bannner[BANNER_NAME] 

    
    #FPS
    cTime = time.time()
    fps =int(1/(cTime-pTime))
    pTime = cTime

    cv.putText(img,msg,MSG_PT, cv.FONT_HERSHEY_COMPLEX,1,C_BLUE,1)
    cv.putText(img,f"FPS:{fps}",FPS_PT, cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
    cv.putText(img,f"github.com/saumilyagupta",GIT_PT, cv.FONT_HERSHEY_COMPLEX_SMALL,1,C_BLACK,1)

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _ , imgInv= cv.threshold(imgGray, 50,255, cv.THRESH_BINARY_INV)
    imgInv= cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)
    # img = cv.bitwise_and(imgCanvas,img)
    # img = cv.bitwise_or(imgCanvas,img)
    # img = cv.addWeighted(img,0.5,imgCanvas,0.5)
    cv.imshow("AIR_DRAW", img)
    cv.waitKey(1)     