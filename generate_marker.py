import cv2 as cv

id = int(input("Input marker id: "))

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

markerImage = cv.aruco.generateImageMarker(arucoDict, id, 200)

cv.imwrite("marker_" + str(id) + ".png", markerImage)
