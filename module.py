import cv2
import time
import estimation as es


cap = cv2.VideoCapture('videos/sample.mp4')
ptime = 0
detector = es.posedetec()

while True:
        success, img = cap.read()
        img = detector.findpose(img)
        lmlist = detector.getposition(img,draw=False)
        if len(lmlist)!=0:
          print(lmlist[0])
          cv2.circle(img, (lmlist[0][1], lmlist[0][2]), 20, (255, 0, 0, 0), cv2.FILLED)

        img = cv2.resize(img, (740, 580))
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (70, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("images", img)
        cv2.waitKey(1)
