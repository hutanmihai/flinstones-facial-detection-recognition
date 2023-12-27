from collections import defaultdict

import cv2 as cv
import numpy as np


def get_images(glob: list[str]) -> list[np.ndarray]:
    return [cv.imread(img) for img in glob]


def get_annotations(path: str) -> dict[str, list[tuple[tuple[int, int, int, int], str]]]:
    dictionary = defaultdict(list)
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(" ")
        image_name: str = line[0]
        xmin: int = int(line[1])
        ymin: int = int(line[2])
        xmax: int = int(line[3])
        ymax: int = int(line[4])
        character: str = line[5].rstrip()
        dictionary[image_name].append(((xmin, ymin, xmax, ymax), character))

    return dictionary
