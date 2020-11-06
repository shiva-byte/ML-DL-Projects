from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import pygame
import requests
import argparse
import math
import imutils
import time
import dlib
import cv2

def avg(lst): 
    return sum(lst) / len(lst)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])	
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 
    
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


pygame.mixer.init()
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
	help=0)
args = vars(ap.parse_args())
 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
MAR_THRESH=0.4
MAR_CONSEC_FRAMES=5
COUNTER1 = 0
COUNTER2 = 0
COUNTER3 =0
vul1=0
vul2=0
vul3=0
fr=0
xl=[]
yl=[]
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nstart, nEnd)= face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mstart, mend)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
print("starting video stream...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        fr += 1
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if fr==1 or fr<=10:
            e1= shape[nstart:nEnd]
            xl.append(e1[7][0])
            yl.append(e1[7][1])
            if fr==10:
                t1=(avg(xl),avg(yl))
        
        leftEye = shape[lStart:lEnd]
        print(leftEye)
        rightEye = shape[rStart:rEnd]
        e = shape[nstart:nEnd]
        mouth= shape[mstart:mend]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar=mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        noseHull = cv2.convexHull(e)
        mouthHull=cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [e], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
       
        
        if vul1 and vul2 and vul3:
             pygame.mixer.Channel(1).play(pygame.mixer.Sound('alarm.wav'))
             cv2.putText(frame, "you are feeling sleepy", (10,30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             url = "https://www.fast2sms.com/dev/bulk"
             payload = "sender_id=FSTSMS&message="your message" &language=english&route=p&numbers=1234567890"
             headers = {'authorization': "fast2sm key",
                        'Content-Type': "application/x-www-form-urlencoded",
                        'Cache-Control': "no-cache",}
             response = requests.request("POST", url, data=payload, headers=headers)
             print(response.text)
             time.sleep(10.0)
            
        if ear < EYE_AR_THRESH:
            COUNTER1 += 1
            if COUNTER1 >= EYE_AR_CONSEC_FRAMES:
                vul3=1
                pygame.mixer.Channel(1).play(pygame.mixer.Sound('eyealrm.wav'))
                cv2.putText(frame, "You are feeling drowsy", (10,30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER1 = 0
            
        if mar >= MAR_THRESH:
            COUNTER2 += 1
            if COUNTER2 >= MAR_CONSEC_FRAMES:
                vul1=1
                pygame.mixer.Channel(1).play(pygame.mixer.Sound('speech.wav'))
                cv2.putText(frame, "You are feeling drowsy", (10,30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER2 = 0
            
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        if fr==48 or fr % 48==0:
            e2=shape[nstart:nEnd]
            t2=tuple(e2[6])
            print("Nose position 1"+str(t1))
            print("Nose position2 after next 48 frames"+str(t2))
        if fr>=48:
            angle=getAngle(t1,(150,350),t2)
            cv2.putText(frame, "Angle: {:.2f}".format(abs(angle)), (30, 80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if abs(angle)>10:
                    vul2=1
                    cv2.putText(frame, "DROWSINESS ALERT! with head tilt", (10, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    pygame.mixer.Channel(0).play(pygame.mixer.Sound('headalrm.wav'))
            else:
                pass
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF    
    
    if key == ord("q"):
        break
pygame.mixer.music.stop()
cv2.destroyAllWindows()
vs.stop()