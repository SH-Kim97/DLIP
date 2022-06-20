import math
import numpy as np
import cv2
from cv2 import *
from cv2.cv2 import *
import time
import mediapipe as mp
import HandTrackingModule as htm
import sudokuMain as sm


pTime = 0

wCam, hCam = 640, 480
cap = VideoCapture(0)
cap.set(CAP_PROP_FRAME_WIDTH, wCam)
cap.set(CAP_PROP_FRAME_HEIGHT, hCam)

moveMode = True
detector = htm.handDetector(maxHands=1, detectionCon=0.75)

buttonArea = [[(280, 90), (360, 150)], [(280, 330), (360, 390)], [(80, 210), (160, 270)], [(480, 210), (560, 270)]] # 0: Up, 1: Down, 2: Left, 3: Right
moveDirection = 0 # 0: None, 1: Up, 2: Down, 3: Left, 4: Right
isClick = False
isMove = False

tipIds = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]

numStack = 0
pNum = 0

numbers, inNumbers, solNumbers, imgSudoku = sm.detectSudoku("1.jpg")
cPos = inNumbers.index(0)
imgResult = imgSudoku.copy()
rightNumbers = [0] * 81
wrongNumbers = [0] * 81

while True:
    _, img = cap.read()
    img = flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if moveMode: # Move mode
        if len(lmList) != 0:
            # Find position of thumb middle point
            if buttonArea[0][0][0] < lmList[3][1] < buttonArea[0][1][0] and buttonArea[0][0][1] < lmList[3][2] < buttonArea[0][1][1]:
                moveDirection = 1
            elif buttonArea[1][0][0] < lmList[3][1] < buttonArea[1][1][0] and buttonArea[1][0][1] < lmList[3][2] < buttonArea[1][1][1]:
                moveDirection = 2
            elif buttonArea[2][0][0] < lmList[3][1] < buttonArea[2][1][0] and buttonArea[2][0][1] < lmList[3][2] < buttonArea[2][1][1]:
                moveDirection = 3
            elif buttonArea[3][0][0] < lmList[3][1] < buttonArea[3][1][0] and buttonArea[3][0][1] < lmList[3][2] < buttonArea[3][1][1]:
                moveDirection = 4
            else:
                moveDirection = 0

            # Check click and move state
            if (not isClick) and lmList[8][2] > lmList[6][2]:
                isClick = True
                isMove = True
            elif (isClick) and lmList[8][2] > lmList[6][2]:
                isMove = False
            else:
                isClick = False
                isMove = False

            # Display direction and click state
            if moveDirection == 1:
                if isMove:
                    img = rectangle(img, buttonArea[0][0], buttonArea[0][1], (0, 255, 0), 3)
                    cPos = cPos - 9
                    if cPos < 0:
                        cPos = cPos + 81
                else:
                    img = rectangle(img, buttonArea[0][0], buttonArea[0][1], (255, 0, 0), 3)
            elif moveDirection == 2:
                if isMove:
                    img = rectangle(img, buttonArea[1][0], buttonArea[1][1], (0, 255, 0), 3)
                    cPos = cPos + 9
                    if cPos > 80:
                        cPos = cPos - 81
                else:
                    img = rectangle(img, buttonArea[1][0], buttonArea[1][1], (255, 0, 0), 3)
            elif moveDirection == 3:
                if isMove:
                    img = rectangle(img, buttonArea[2][0], buttonArea[2][1], (0, 255, 0), 3)
                    cPos = cPos - 1
                    if cPos % 9 == 8:
                        cPos = cPos + 9
                else:
                    img = rectangle(img, buttonArea[2][0], buttonArea[2][1], (255, 0, 0), 3)
            elif moveDirection == 4:
                if isMove:
                    img = rectangle(img, buttonArea[3][0], buttonArea[3][1], (0, 255, 0), 3)
                    cPos = cPos + 1
                    if cPos % 9 == 0:
                        cPos = cPos - 9
                else:
                    img = rectangle(img, buttonArea[3][0], buttonArea[3][1], (255, 0, 0), 3)
        
        putText(img, "Move Mode", (20, 40), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    else: # Solve mode
        # Detect fingers
        fingers = []
        if len(lmList) != 0:
            for id, cx, cy in lmList[:]:
                if id in tipIds:
                    if id in [4, 25]: # Thumb
                        if lmList[id - 1][1] < cx < lmList[id + 13][1] or lmList[id + 13][1] < cx < lmList[id - 1][1]:
                            fingers.append(0)
                        else:
                            fingers.append(1)

                    else: # 4 Fingers
                        if cy < lmList[id - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

            totalFingers = fingers.count(1) # Count fingers

            # Stack same number of fingers
            if pNum == totalFingers:
                numStack += 1
            else:
                numStack = 0
            pNum = totalFingers
            
            # Input the number when number of fingers maintained for 20 frame
            if numStack == 20 and totalFingers != 10 and inNumbers[cPos] != -1:
                inNumbers[cPos] = totalFingers
                try:
                    cPos = inNumbers.index(0)
                except:
                    pass

            # Print number of fingers
            rectangle(img, (0, 420), (80, 480), (0, 255, 0), FILLED)
            putText(img, str(totalFingers), (20, 460), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
        
        putText(img, "Solve Mode", (20, 40), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # Display sudoku image
    Sudoku = imgSudoku.copy()
    Sudoku = sm.inNumbers(Sudoku, inNumbers)
    Sudoku = sm.selectedGrid(Sudoku, cPos)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    putText(img, f'FPS: {int(fps)}', (500, 40), FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    imshow('Sudoku', Sudoku)
    imshow("Controller", img)

    inkey = waitKey(1)
    # Change mode
    if inkey == 77 or inkey == 109:
        moveMode = not moveMode
        detector = htm.handDetector(maxHands=1+(not moveMode), detectionCon=0.75)
    # Finish and display result
    elif inkey == 70 or inkey == 102:
        if inNumbers == solNumbers:
            imgResult = sm.inNumbers(imgResult, solNumbers, color = (0, 255, 0))
            putText(Sudoku, "Success!", (50, 240), FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        else:
            for i in range(81):
                if solNumbers[i] == inNumbers[i]:
                    rightNumbers[i] = solNumbers[i]
                else:
                    wrongNumbers[i] = solNumbers[i]
            imgResult = sm.inNumbers(imgResult, rightNumbers, color = (0, 255, 0))
            imgResult = sm.inNumbers(imgResult, wrongNumbers, color = (0, 0, 255))
            putText(Sudoku, "Fail!", (150, 240), FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
        imshow('Result', imgResult)
        imshow('Sudoku', Sudoku)
        waitKey()
        break
    # Break
    elif inkey == 27:
        break

destroyAllWindows()