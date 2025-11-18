import cv2 as cv
import numpy as np

camIndex = int((input("Input cam index (0 - iPhone, 1 - Mac), (default: 0): ")) or 0)
tabCount = int((input("Input tab amount (default: 1): ")) or 1)


def checkTab(tabIndex, ids, frame):
    tabOffset = tabIndex * 4
    try:
        idx_0 = np.where(ids == 0 + tabOffset)[0][0]
        idx_1 = np.where(ids == 1 + tabOffset)[0][0]
        idx_2 = np.where(ids == 2 + tabOffset)[0][0]
        idx_3 = np.where(ids == 3 + tabOffset)[0][0]

        c0 = corners[idx_0][0][0]
        c1 = corners[idx_1][0][1]
        c2 = corners[idx_2][0][2]
        c3 = corners[idx_3][0][3]

        srcPoints = np.array([c0, c1, c2, c3], dtype="float32")
        dstPoints = np.array(
            [
                [0, 0],  # top left
                [tabSizeX, 0],  # top right
                [tabSizeX, tabSizeY],  # bottom right
                [0, tabSizeY],  # bottom left
            ],
            dtype="float32",
        )

        mat = cv.getPerspectiveTransform(srcPoints, dstPoints)
        warped = cv.warpPerspective(frame, mat, (tabSizeX, tabSizeY))

        cv.imshow(f"tablica {tabIndex}", warped)
        cv.polylines(frame, [np.int32(srcPoints)], True, (0, 0, 255), 4)
    except IndexError:
        pass


cam = cv.VideoCapture(camIndex)

if not cam.isOpened():
    print(f"Can't open camera {camIndex}")

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

tabSizeX, tabSizeY = 640, 640

while True:
    ret, frame = cam.read()
    if not ret:
        print(f"Couldn't capture frame {frame}")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()
        for i in range(0, tabCount):
            checkTab(i, ids, gray)

    cv.imshow("Camera", frame)
    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()
