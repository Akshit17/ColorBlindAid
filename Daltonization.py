import cv2
import numpy as np
import argparse
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
    editablePhoto = getImageArray(RGBConvert, editablePhoto, rowx, coly)
    for i in range(0, rowx):
        for j in range(0, coly):
            for k in range(0, 3):
                editablePhoto[i, j, k] = ((editablePhoto[i, j, k])) * 255

    NormalPhoto = normalise(editablePhoto, rowx, coly)
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
    protanopeConvert = np.array(
        [[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]
    )  # correction filter array for protonopia
    editablePhoto = getImageArray(protanopeConvert, editablePhoto, rowx, coly)
    NormalPhoto = normalise(editablePhoto, rowx, coly)
    return NormalPhoto


# Simulating Deutranopia
def ConvertToDeuteranopes(editablePhoto, rowx, coly):
    DeuteranopesConvert = np.array(
        [[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]
    )  # correction filter array for deutranopia
    editablePhoto = getImageArray(DeuteranopesConvert, editablePhoto, rowx, coly)
    NormalPhoto = normalise(editablePhoto, rowx, coly)
    return NormalPhoto


# Simulating Tritanopia
def ConvertToTritanope(editablePhoto, sizeX, sizeY):
    TritanopeConvert = np.array(
        [[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]
    )  # correction filter array for tritanopia
    editablePhoto = getImageArray(TritanopeConvert, editablePhoto, sizeX, sizeY)
    NormalPhoto = normalise(editablePhoto, sizeX, sizeY)
    return NormalPhoto


def arrayToImage(editablePhoto, rowx, coly, saveAs):
    rgbArray = np.zeros((rowx, coly, 3), "uint8")
    for i in range(0, rowx):
        for j in range(0, coly):
            for k in range(0, 3):
                rgbArray[i, j, k] = editablePhoto[i, j, k]
    img = Image.fromarray(rgbArray)
    img.save(saveAs)


def daltonize(originalRgb, simRgb, rowx, coly):
    photo = originalRgb.read()
    editablePhoto = np.zeros((rowx, coly, 3), "float")
    for i in range(0, rowx):
        for j in range(0, coly):
            for k in range(0, 3):
                editablePhoto[i, j, k] = photo[i, j][k]

    diffPhoto = simRgb - editablePhoto
    transMatrix = numpy.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    errCorrection = getImageArray(transMatrix, diffPhoto, rowx, coly)
    finalImage = errCorrection + editablePhoto
    return finalImage


def main():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
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
        
        
        #frame = Image.open("sample.jpg")  USING THIS WORKS and gives a single image saved on device as the output. 
        # Use outside the while funtion
        
        lmsPhoto = tolms(frame, rowx, coly)  # converting to lms

        simPhoto = ConvertToProtanopes(lmsPhoto, rowx, coly)
        # simPhoto = ConvertToDeuteranopes(lmsPhoto,rowx,coly)
        # simPhoto = ConvertToTritanope(lmsPhoto,rowx,coly)

        rgbPhoto = convertToRGB(simPhoto, rowx, coly)
        rgbPhoto = daltonize(inputIm, rgbPhoto, rowx, coly)
        arrayToImage(rgbPhoto, rowx, coly, "outImage_RG" + str(4) + ".jpg")

        cv2.imshow("Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # stream ends/closes on pressing  q
            break

    # release the camera capture cap so if a new camera capture cap2 is created then it can takeover

    cap.release()

    cv2.destroyAllWindows()
