import cv2 as cv

startId = int(input("Input first marker id: "))
endId = int(input("Input last marker id: "))

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

for i in range(startId, endId + 1):
    markerImage = cv.aruco.generateImageMarker(arucoDict, i, 200)
    cv.imwrite("marker_" + str(i) + ".png", markerImage)
