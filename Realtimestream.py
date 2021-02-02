import cv2
import numpy as np


def livestreamc(frame):
{ 
    cap = cv2.VideoCapture(0)
    while True:
    ret, frame = cap.read()
  
     return frame

}
 
 
 
 # cv2.imshow("Name", frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # cv2.imshow("Grayed", gray)

   #if cv2.waitKey(1) & 0xFF == ord("q"):
    #   break



# release the camera capture cap so if a new camera capture cap2 is created then it can takeover
#cap.release()
#cv2.destroyAllWindows()