import numpy as np
import cv2 as cv
import os

from src.constants import (
    POSITIVES_PATH,
    NEGATIVES_PATH,
    MIN_WIDTH,
    MAX_WIDTH,
    MIN_HEIGHT,
    MAX_HEIGHT,
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
            cv.imwrite(f"{POSITIVES_PATH}/{image_index}_{counter}.jpg", image[ymin:ymax, xmin:xmax])
            counter += 1


def extract_negatives(images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]) -> None:
    # TODO: REFACTOR THIS, IT IS ERROR PRONE TO RANGE RANDOM ERROR
    for image_index in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_index)]
        coordinates = [coord for coord, _ in annotations[image_index]]

        while counter < len(coordinates) * 2:  # * 2 because we will flip the positives for more data
            width = randint(MIN_WIDTH, MAX_WIDTH - 100)
            min_height = max(int(width * 0.8), MIN_HEIGHT)
            max_height = min(int(width * 1.2), MAX_HEIGHT)
            height = randint(min_height, max_height)

            x = randint(0, MAX_WIDTH - width)
            y = randint(0, MAX_HEIGHT - height)
            box = (x, y, x + width, y + height)
            if check_overlap(box, coordinates):
                cv.imwrite(f"{NEGATIVES_PATH}/{image_index}_{counter}.jpg", image[y : y + height, x : x + width])
                counter += 1


def extract_positives_and_negatives(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist()
    extract_positives(images, annotations)
    extract_negatives(images, annotations)
