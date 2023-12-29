import numpy as np
import cv2 as cv
import os

from src.constants import (
    POSITIVES_PATH,
    NEGATIVES_PATH,
    MAX_WIDTH,
    MAX_HEIGHT,
    POS_NEG_WIDTH,
    POS_NEG_HEIGHT,
)
from random import randint


def check_if_dirs_exist():
    if not os.path.exists(POSITIVES_PATH):
        os.makedirs(POSITIVES_PATH)

    if not os.path.exists(NEGATIVES_PATH):
        os.makedirs(NEGATIVES_PATH)


def check_overlap(box, coordinates):
    xmin, ymin, xmax, ymax = box
    for coord in coordinates:
        cxmin, cymin, cxmax, cymax = coord
        if not (xmax < cxmin or xmin > cxmax or ymax < cymin or ymin > cymax):
            return False
    return True


def extract_positives(images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]) -> None:
    for image_index in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_index)]
        for coordinates, character in annotations[image_index]:
            xmin, ymin, xmax, ymax = coordinates
            box = cv.resize(image[ymin:ymax, xmin:xmax], (POS_NEG_WIDTH, POS_NEG_HEIGHT))
            cv.imwrite(f"{POSITIVES_PATH}/{str(image_index).zfill(4)}_{counter}.jpg", box)
            counter += 1


def extract_negatives(images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]) -> None:
    # TODO: REFACTOR THIS, IT IS ERROR PRONE TO RANGE RANDOM ERROR
    for image_index in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_index)]
        coordinates = [coord for coord, _ in annotations[image_index]]

        while counter < len(coordinates) * 2:  # * 2 because we will flip the positives for more data
            x = randint(0, MAX_WIDTH - POS_NEG_WIDTH)
            y = randint(0, MAX_HEIGHT - POS_NEG_HEIGHT)
            box = (x, y, x + POS_NEG_WIDTH, y + POS_NEG_HEIGHT)
            if check_overlap(box, coordinates):
                cv.imwrite(
                    f"{NEGATIVES_PATH}/{str(image_index).zfill(4)}_{counter}.jpg".zfill(5),
                    image[y : y + POS_NEG_HEIGHT, x : x + POS_NEG_WIDTH],
                )
                counter += 1


def extract_positives_and_negatives(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist()
    extract_positives(images, annotations)
    extract_negatives(images, annotations)
