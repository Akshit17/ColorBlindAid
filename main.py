# import fastapi   this does not work

from fastapi import FastAPI  # this works
from fastapi.responses import StreamingResponse
import cv2
import numpy as np

import uvicorn  # not needed?

app = FastAPI()  # calling FastAPI class from fastapi

cap = cv2.VideoCapture(0)


@app.get("/urlname")  # "urlname" added after / for unique address
async def root(x=2, y=9):
 return x + y


@app.post("/api/predict")
def stream_frame(file: Uploadfile = File(...)):
    {
          image = livestremc(file)

    }
    #return StreamingResponse(frame)


while True:
    ret, frame = cap.read()
    cv2.imshow('Name.png', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayed", gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# waits for one second to take any keyboard key input , in this case 'q' only allowed
# release the camera capture cap so if a new camera capture cap2 is created then it can takeover
cap.release()
cv2.destroyAllWindows()

# if __name__  == "__main__":
# uvicorn.run(app,port=8000,host='0.0.0.0')
