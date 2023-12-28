import numpy as np
from src.utils.collapse import collapse
from src.constants import COLLAPSED_ANNOTATIONS_PATH, COLLAPSED_NUMPY_PATH, COLOR_CHARACTER_MAPPING
from src.utils.helpers import show_image
from src.utils.readers import get_annotations
import cv2 as cv

if __name__ == "__main__":
    # collapse()

    # Initialize the annotations and images
    train_images = np.load(COLLAPSED_NUMPY_PATH)
    annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)

    for image_index in annotations.keys():
        image: np.ndarray = train_images[int(image_index)]
        for coordinates, character in annotations[image_index]:
            xmin, ymin, xmax, ymax = coordinates
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 3)
        show_image(image, image_index)
