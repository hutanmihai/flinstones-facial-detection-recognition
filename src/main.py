import numpy as np
from src.utils.collapse import collapse
from src.constants import COLLAPSED_ANNOTATIONS_PATH, COLLAPSED_NUMPY_PATH, COLOR_CHARACTER_MAPPING
from src.utils.generate_positives_negatives import extract_positives_and_negatives
from src.utils.helpers import show_image
from src.utils.readers import get_annotations

if __name__ == "__main__":
    # Collapse the images and annotations
    # collapse()

    # Initialize the annotations and images
    train_images = np.load(COLLAPSED_NUMPY_PATH)
    annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)

    # Generate the positives and negatives
    extract_positives_and_negatives(train_images, annotations)

    # for image_index in annotations.keys():
    #     image: np.ndarray = train_images[int(image_index)]
    #     for coordinates, character in annotations[image_index]:
    #         xmin, ymin, xmax, ymax = coordinates
    #         cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 3)
    #     show_image(image, image_index)
