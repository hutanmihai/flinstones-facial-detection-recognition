import pickle

import numpy as np
import cv2 as cv
from skimage.feature import hog
from sklearn.svm import LinearSVC

from src.constants import POSITIVES_GLOB, PIXELS_PER_CELL, CELLS_PER_BLOCK, ORIENTATIONS, NEGATIVES_GLOB, MODEL_PATH
from src.utils.readers import get_images


def get_positive_descriptors():
    images = get_images(POSITIVES_GLOB)
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(
            image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS
        )
        descriptors.append(features)

        features = hog(
            np.fliplr(image),
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            orientations=ORIENTATIONS,
        )
        descriptors.append(features)

    descriptors = np.array(descriptors)
    return descriptors


def get_negatives_descriptors():
    images = get_images(NEGATIVES_GLOB)
    print(len(images))
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(
            image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS
        )
        descriptors.append(features)

    descriptors = np.array(descriptors)
    return descriptors


def train_classifier():
    positive_features = get_positive_descriptors()
    negative_features = get_negatives_descriptors()
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    model = LinearSVC(dual=True)
    model.fit(training_examples, train_labels)
    acc = model.score(training_examples, train_labels)
    print(f"Accuracy: {acc}")
    pickle.dump(model, open(MODEL_PATH / "model.pkl", "wb"))
    scores = model.decision_function(training_examples)

    # Visualize how are the scores distributed for positive and negative examples
    positive_scores = scores[train_labels > 0]
    negative_scores = scores[train_labels <= 0]
    print(f"Positive scores: {positive_scores}")
    print(f"Negative scores: {negative_scores}")
