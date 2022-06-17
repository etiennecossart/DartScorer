# -*- coding: utf-8 -*-

from turtle import width
import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle
 
cap = cv2.VideoCapture(0)
boarderCood_org = [[615,210], [1285,245], [580, 860], [1255, 890]] # Arm rest camera angle - June 8, 2022 (threshold corner)

colorFinder = ColorFinder(False)  
       
hsvVals = {'hmin': 100, 'smin': 175, 'vmin': 39, 'hmax': 143, 'smax': 237, 'vmax': 242}



def getBoard(img):
    width, height = int(650 * 1.25), int(550 * 1.25)
    pts1 = np.float32(boarderCood_org)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imageOutput = cv2.warpPerspective(img, matrix, (width, height))
    for x in range(len(boarderCood_org)):
        cv2.circle(img, (boarderCood_org[x][0], boarderCood_org[x][1]), 5, (0,255,0),cv2.FILLED)
    return imageOutput


def detectColorDarts(imgToMask):
    imgBlur = cv2.GaussianBlur(imgToMask, (3,3),2) #adding blur to img, GaussianBlur(img, odd number for kernel size, delta value)
    imgColor, msk = colorFinder.update(imgBlur, hsvVals)
    kernel = np.ones((11, 11), np.uint8)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("ImageColor", imgColor)
    return msk




## takes in video feed from camera
while True: 
    # # *** Resetting camera *** 
    # success, img = cap.read()
    # imgBoard_callibration = resetCamera_getBoard(img)
    # cv2.imshow("Callibrator", img)
    # cv2.imshow("Dart Board", imgBoard_callibration)
    # cv2.waitKey(1)

    # # *** Writing img to file ***
    # success, img = cap.read()
    # imgBoard = getBoard(img)
    # cv2.imwrite('BlueDart.png', imgBoard)
    # cv2.waitKey(1)

    # # *** Revise Mask - Set colorFinder param to True ***
    # success, img = cap.read()
    # imgBoard = getBoard(img)
    # mask = detectColorDarts(imgBoard)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(1)


    success, img = cap.read()
    imgBoard = getBoard(img)
    # cv2.imshow("imgBoard 1", imgBoard) # Showing imgBoard
    mask = detectColorDarts(imgBoard)
    # cv2.imshow("mask", mask) # Showing Binary Mask
    imgContours, conFound = cvzone.findContours(imgBoard, mask, 800)
    # cv2.imshow("Contours", imgContours) # Showing filtered color
    


    img_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    im2,contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = contours[0]
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
    cv2.waitKey(1)


