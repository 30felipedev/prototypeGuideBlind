import cv2
import numpy as np
import utils
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model  
import cv2  
import numpy as np
from cvzone.SerialModule import SerialObject
import utils
ard = SerialObject('COM3')
np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
curveList = []
avgVal = 10
def getLaneCurve(img,display=2):
    imgCopy = img.copy()
    imgResult = img.copy()
    imgThres = utils.thresholding(img)
    hT,wT,c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres,points,wT,hT)
    imgWarpPoints = utils.drawPoints(imgCopy,points)
    midPoint,imgHist = utils.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
    curveAvgpoint,imgHist = utils.getHistogram(imgWarp,display=True,minPer=0.9,region=1)
    curveRaw = curveAvgpoint-midPoint

    curveList.append(curveRaw)
    if len(curveList)>curveAvgpoint:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    if display != 0:
            imgInvWarp = utils.warpImg(imgWarp, points, wT, hT,inv = True)
            imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
            imgInvWarp[0:hT//3,0:wT] = 0,0,0
            imgLaneColor = np.zeros_like(img)
            imgLaneColor[:] = 0, 255, 0
            imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
            imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
            midY = 450
            cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
            cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
            cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       
            for x in range(-30, 30):
                w = wT // 20
                cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
            #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3)
    
    if display == 2:
        imgStacked = utils.stackImages(0.7,([img,imgWarpPoints,imgWarp],[imgHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt',imgResult)
    
    return curve

cap = cv2.VideoCapture('vid1.mp4')
initialTrackBarVals = [102,80,20,214]
utils.initializeTrackbars(initialTrackBarVals)
frameCounter = 0

while True:
    _, img = cap.read()
    img = cv2.resize(img,(480,240)) 
    curve = getLaneCurve(img,display=2)
    print(curve)
    if curve > 1:
         ard.sendData([1,0])
    if curve < 1:
         ard.sendData([0,1])
    if curve == 0:
         ard.sendData([0,0])
          
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break