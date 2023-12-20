import cv2 as cv
import numpy as np


def get_images(glob: list[str]) -> list[np.ndarray]:
    return [cv.imread(img) for img in glob]


def get_annotations(path: str) -> dict[str, tuple[int, int, int, int], str]:
    with open(path, "r") as f:
        return f.readlines()
