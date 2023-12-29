import pickle

import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from src.constants import (
    COLLAPSED_ANNOTATIONS_PATH,
    COLLAPSED_NUMPY_PATH,
    COLOR_CHARACTER_MAPPING,
    NEGATIVES_GLOB,
    POSITIVES_GLOB,
    VALIDATION_ANNOTATIONS_PATH,
    VALIDATION_IMAGES,
    VALIDATION_IMAGES_PATH,
    VALIDATION_NUMPY_PATH,
)
from src.utils.collapse import collapse, save_train_images_numpy, save_validation_images_numpy
from src.utils.generate_positives_negatives import (
    extract_positives_and_negatives,
    extract_positives_and_negatives_validation,
)
from src.utils.helpers import show_image
from src.utils.readers import get_annotations, get_images


def get_positive_descriptors():
    images = get_images(POSITIVES_GLOB)
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_RGB2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(image, feature_vector=True)
        descriptors.append(features)

        features = hog(np.fliplr(image), feature_vector=True)
        descriptors.append(features)

    descriptors = np.array(descriptors)
    return descriptors


def get_negatives_descriptors():
    images = get_images(NEGATIVES_GLOB)
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(image, feature_vector=False)
        descriptors.append(features.flatten())

    descriptors = np.array(descriptors)
    return descriptors


# def train_classifier():
#     positive_features = get_positive_descriptors()
#     negative_features = get_negatives_descriptors()
#     training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
#     train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
#     model = LinearSVC()
#     model.fit(training_examples, train_labels)
#     acc = model.score(training_examples, train_labels)
#     print(f"Accuracy: {acc}")
#     validationn_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in get_images(VALIDATION_IMAGES)]
#     validation_examples = np.concatenate((np.squeeze(validationn_images), np.squeeze(negative_features)), axis=0)
#     predicted_labels = model.predict(validation_examples)
#     true_labels = np.concatenate((np.ones(len(validation_images)), np.zeros(len(negative_features))))
#     print(accuracy_score(true_labels, predicted_labels))


if __name__ == "__main__":
    # Collapse the images and annotations, save numpys
    # save_train_images_numpy()
    # save_validation_images_numpy()
    # collapse()

    # Initialize the annotations and images
    train_images = np.load(COLLAPSED_NUMPY_PATH)
    annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)

    validation_images = np.load(VALIDATION_NUMPY_PATH)
    validation_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)

    # Generate the positives and negatives
    # extract_positives_and_negatives(train_images, annotations)
    # extract_positives_and_negatives_validation(validation_images, validation_annotations)

    # Train the classifier
    # train_classifier()

    # for image_index in annotations.keys():
    #     image: np.ndarray = train_images[int(image_index)]
    #     for coordinates, character in annotations[image_index]:
    #         xmin, ymin, xmax, ymax = coordinates
    #         cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 1)
    #     # if int(image_index) in [1261, 3639, 3670]:
    #     show_image(image, image_index)

    # for image_index in validation_annotations.keys():
    #     image: np.ndarray = validation_images[int(image_index)]
    #     for coordinates, character in validation_annotations[image_index]:
    #         xmin, ymin, xmax, ymax = coordinates
    #         cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 1)
    #     show_image(image, image_index)
