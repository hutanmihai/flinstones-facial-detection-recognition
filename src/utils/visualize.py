import cv2 as cv
import numpy as np

from src.constants import COLOR_CHARACTER_MAPPING
from src.utils.helpers import show_image


def visualize_images_with_boxes(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
):
    for image_index in annotations.keys():
        image: np.ndarray = images[int(image_index)]
        for coordinates, character in annotations[image_index]:
            xmin, ymin, xmax, ymax = coordinates
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 1)
        show_image(image, image_index)
