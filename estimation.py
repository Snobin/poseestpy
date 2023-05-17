import cv2
import mediapipe as mp
import time

class posedetec():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon > 0.0  # Convert to bool
        self.trackCon = trackCon > 0.0 
        self.mppose = mp.solutions.pose
        self.mpdraw = mp.solutions.drawing_utils
        self.pose = self.mppose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findpose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks and draw:
            self.mpdraw.draw_landmarks(img, results.pose_landmarks,
                                       self.mppose.POSE_CONNECTIONS)
        return img
    
    def getposition(self, img, draw=True):
        lmlist = []
        results = self.pose.process(img)  # Corrected variable name
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):  # Corrected variable name
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0, 0), cv2.FILLED)
        return lmlist   

def main():
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
