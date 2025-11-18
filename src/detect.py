import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Frame:
    id: int
    data: MatLike


def checkTab(corners, tabIndex, ids, frame, tabSize, padding) -> Optional[MatLike]:
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
                [padding, padding],  # top left
                [tabSize[0] + padding, padding],  # top right
                [tabSize[0] + padding, tabSize[1] + padding],  # bottom right
                [padding, tabSize[1] + padding],  # bottom left
            ],
            dtype="float32",
        )

        mat = cv.getPerspectiveTransform(srcPoints, dstPoints)
        warped = cv.warpPerspective(
            frame, mat, (tabSize[0] + 2 * padding, tabSize[1] + 2 * padding)
        )
        return warped
    except IndexError:
        return None


def getInput(frame, detector, tabCount, tabSize, padding) -> List[Frame]:
    frames = []

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()
        for i in range(0, tabCount):
            newData = checkTab(corners, i, ids, frame, tabSize, padding)
            if newData is not None:
                frames.append(Frame(i, newData))

    return frames
