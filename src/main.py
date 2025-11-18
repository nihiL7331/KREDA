import cv2 as cv
from detect import getInput

camIndex = int(
    (input("Input cam index (0 - iPhone, 1 - Mac), (default: 0): ")) or int(0)
)
tabCount = int((input("Input tab amount (default: 1): ")) or int(1))

tabSizeX, tabSizeY = 640, 640
padding = 0


def main():
    cam = cv.VideoCapture(camIndex)

    if not cam.isOpened():
        print(f"Can't open camera {camIndex}")

    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"Couldn't capture frame {frame}")
            continue
        frames = getInput(frame, detector, tabCount, [tabSizeX, tabSizeY], padding)
        for frame in frames:
            cv.imshow(f"frame{frame.id}", frame.data)
        if cv.waitKey(1) == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
