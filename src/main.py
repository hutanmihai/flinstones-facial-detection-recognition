from src.constants import (
    VALIDATION_IMAGES_PATH,
    VALIDATION_ANNOTATIONS_PATH,
    COLLAPSED_IMAGES_PATH,
    COLLAPSED_ANNOTATIONS_PATH,
)
from src.task1_hog import run_task1_hog
from src.train_hog_classifier import train_hog_classifier
from src.utils.collapse import collapse
from src.utils.generate_positives_negatives import extract_train_and_validation_patches
from src.utils.visualize import visualize_images_with_boxes, visualize_images_with_boxes_and_detections

if __name__ == "__main__":
    # Collapse the images and annotations
    collapse()
    # Extract positives and negatives for training and validation
    extract_train_and_validation_patches()

    # visualize_images_with_boxes(COLLAPSED_IMAGES_PATH, COLLAPSED_ANNOTATIONS_PATH)
    # visualize_images_with_boxes(VALIDATION_IMAGES_PATH, VALIDATION_ANNOTATIONS_PATH)

    train_hog_classifier()
    run_task1_hog()

    # visualize_images_with_boxes_and_detections(VALIDATION_ANNOTATIONS_PATH)
