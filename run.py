# version 1.0.3
from collections import deque
import multiprocessing as mp
import _thread as thr
import numpy as np
import os, time
import cv2

import prediction as pre
import get_video as video


FLAG_FALL = False
VIDEO_GETTING = False
FLAG_RUNNING_STATUS = False

camera_ip_l = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frameNum = 0
fall_cancel = 0.4
data = deque(maxlen=50)
x, y, w, h = 0, 0, 0, 0
a, area, motion_speed, tempY = 0, 0, 0, 0
fall_count = 0

def do_analysis():
    global frameNum
    global x, y, w, h
    global a, area, motion_speed, tempY
    global fall_count

    while True:
        ret, frame = cap.read()
        frameNum += 1
        tempframe = frame.copy()

        def fall_or_not():
            global FLAG_FALL
            global VIDEO_GETTING
            if (frameNum % 5 == 0):
                if True:
                    res = pre.prediction(data)
                    # print(res)
                    if (res[0] == 0 and res[1] > 0.92):
                        FLAG_FALL = True
                        print('FLAG_FALL')
                        cv2.putText(tempframe, 'Status: Fall', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    elif (res[0] == 2 and res[1] > fall_cancel):
                        FLAG_FALL = False
                        print('Nomal')
                        cv2.putText(tempframe, 'Status: Nomal', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        print('Nomal')
                        cv2.putText(tempframe, 'Status: Nomal', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)




        if (frameNum == 1):
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)

        if (frameNum >= 2):
            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            currentframe = cv2.dilate(currentframe, None, iterations=3)
            currentframe = cv2.erode(currentframe, None, iterations=2)
            ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 1300 or cv2.contourArea(c) > 19200:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)  # update the rectangle
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)

            if (w*h == 0):
                pass

            ys = y-tempY
            ys = ys if (ys < 16) else 1
            tempY = y
            motion_speed = area/500*ys
            motion_speed = 0 if (abs(motion_speed) >= 120) else motion_speed
            data.append([y-30, (w/(h+0.1)-0.5)*100, motion_speed])

            if len(data) >= 50:
                try:
                    thr.start_new_thread(fall_or_not, ())
                except:
                    print("Error: cannot start the thread")

        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
        cv2.imshow('window_camera', tempframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def run():
    do_analysis()


if __name__ == '__main__':
    run()