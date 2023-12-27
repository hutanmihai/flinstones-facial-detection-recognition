import numpy as np
import cv2 as cv
import os

from src.utils.helpers import show_image
from src.utils.readers import get_images, get_annotations
from src.constants import (
    BARNEY_ANNOTATIONS_PATH,
    BETTY_ANNOTATIONS_PATH,
    FRED_ANNOTATIONS_PATH,
    WILMA_ANNOTATIONS_PATH,
    POSITIVES_PATH,
    NEGATIVES_PATH,
    BARNEY_IMAGES,
)


def check_if_dirs_exist():
    if not os.path.exists(POSITIVES_PATH):
        os.makedirs(POSITIVES_PATH)

    if not os.path.exists(NEGATIVES_PATH):
        os.makedirs(NEGATIVES_PATH)


if __name__ == "__main__":
    check_if_dirs_exist()
