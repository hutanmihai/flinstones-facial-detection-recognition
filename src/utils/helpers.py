import os
from pathlib import Path

import cv2 as cv
import numpy as np

########################################################################################################################
# CORE SHOW IMAGE FUNCTION
########################################################################################################################


def show_image(image: np.ndarray, title: str = "image") -> None:
    """
    Shows the image.
    :param title: the title of the window, default is "image"
    :param image: the image to show
    :return:
    """
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def check_if_dirs_exist(paths: list[Path]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
