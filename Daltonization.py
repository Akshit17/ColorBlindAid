import cv2
import numpy as np
import argparse
import sys
import colorsys


def getImageArray(respectiveArray, editablePhoto, rowx, coly):
    for i in range(0, rowx):
        for j in range(0, coly):
            currMatrix = np.array((0, 0, 0), dtype=float)
            for k in range(0, 3):
                currMatrix[k] = editablePhoto[i, j, k]
            lmsImage = np.dot(respectiveArray, currMatrix)
            for k in range(0, 3):
                editablePhoto[i, j, k] = lmsImage[k]
    return editablePhoto


def tolms(frame, rowx, coly):

    photo = cv2.imread(frame)
    editablePhoto = np.zeros((rowx, coly, 3), "float")

    for i in range(0, sizeX):
        for j in range(0, sizeY):
            for k in range(0, 3):
                editablePhoto[i, j, k] = photo[i, j][k]
                editablePhoto[i, j, k] = (editablePhoto[i, j, k]) / 255
    lmsConvert = np.arraynp.array(
        (
            [
                [17.8824, 43.5161, 4.11935],
                [3.45565, 27.1554, 3.86714],
                [0.0299566, 0.184309, 1.46709],
            ]
        )
    )
    editablePhoto = getImageArray(lmsConvert, editablePhoto, rowx, coly)
    NormalPhoto = normalise(editablePhoto, rowx, coly)
    return NormalPhoto


def convertToRGB(editablePhoto, rowx, coly):
    rgb2lms = numpy.array(
        [
            [17.8824, 43.5161, 4.11935],
            [3.45565, 27.1554, 3.86714],
            [0.0299566, 0.184309, 1.46709],
        ]
    )
    RGBConvert = numpy.linalg.inv(rgb2lms)
    # print(RGBConvert)
    editablePhoto = getImageArray(RGBConvert, editablePhoto, sizeX, sizeY)
    for i in range(0, rowx):
        for j in range(0, coly):
            for k in range(0, 3):
                editablePhoto[i, j, k] = ((editablePhoto[i, j, k])) * 255

    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto


def normalise(editablePhoto, rowx, coly):
    NormalPhoto = np.zeros((rowx, coly, 3), "float")
    x = rowx - 1
    y = coly
    for i in range(0, rowx):
        for j in range(0, coly):
            for k in range(0, 3):
                NormalPhoto[x, j, k] = editablePhoto[i, j, k]
        x = x - 1

    return NormalPhoto


# Simulating for protanopes
def ConvertToProtanopes(editablePhoto, rowx, coly):
    protanopeConvert = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
    editablePhoto = getImageArray(protanopeConvert, editablePhoto, rowx, coly)
    NormalPhoto = normalise(editablePhoto, rowx, coly)
    return NormalPhoto


# Simulating Deutranopia
def ConvertToDeuteranopes(editablePhoto, sizeX, sizeY):
    DeuteranopesConvert = numpy.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
    editablePhoto = getImageArray(DeuteranopesConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto


# Simulating Tritanopia
def ConvertToTritanope(editablePhoto, sizeX, sizeY):
    TritanopeConvert = numpy.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
    editablePhoto = getImageArray(TritanopeConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto


def arrayToImage(editablePhoto, sizeX, sizeY, saveAs):
    rgbArray = np.zeros((sizeX, sizeY, 3), "uint8")
    for i in range(0, sizeX):
        for j in range(0, sizeY):
            for k in range(0, 3):
                rgbArray[i, j, k] = editablePhoto[i, j, k]
    img = Image.fromarray(rgbArray)
    img.save(saveAs)


def daltonize(originalRgb, simRgb, sizeX, sizeY):
    photo = originalRgb.load()
    editablePhoto = np.zeros((sizeX, sizeY, 3), "float")
    for i in range(0, sizeX):
        for j in range(0, sizeY):
            for k in range(0, 3):
                editablePhoto[i, j, k] = photo[i, j][k]

    diffPhoto = simRgb - editablePhoto
    transMatrix = numpy.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    errCorrection = getImageArray(transMatrix, diffPhoto, sizeX, sizeY)
    finalImage = errCorrection + editablePhoto
    return finalImage


def main():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow("Stream", frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Grayed", gray)

        # B,G,R = cv2.split(frame) Spliting the channels like this caused blacking out of output stream
        # which i was not able to fix
        rowx = frame.shape[0]
        coly = frame.shape[1]

        # zeros = np.zeros(frame.shape[:2], dtype="uint8")  #Creates a of width*height pixels carrying zero
        # print(zeros)

        # Redchannel = cv2.merge([zeros, zeros, R])
        # cv2.imshow("Red", Redchannel)

        # framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb3darray = np.array(framergb)
        lmsPhoto = tolms(frame, rowx, coly)  # converting to lms

        # r = rgb3darray[479][639][0]
        # g = rgb3darray[479][639][1]   trying to change pixel values of rgb in the 3d array
        # b = rgb3darray[479][639][2]
        # print(r)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # release the camera capture cap so if a new camera capture cap2 is created then it can takeover

    cap.release()

    cv2.destroyAllWindows()
