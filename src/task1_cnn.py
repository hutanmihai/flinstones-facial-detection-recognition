import timeit

import cv2 as cv
import numpy as np
import torch
from skimage.transform import resize
from torchvision.transforms import transforms

from src.constants import (
    VALIDATION_NUMPY_PATH,
    THRESHOLD,
    DIM_HOG_WINDOW,
    MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SOLUTION_DETECTIONS_PATH,
    SOLUTION_SCORES_PATH,
    SOLUTION_FILE_NAMES_PATH,
)
from src.utils.generate_positives_negatives import (
    extract_cnn_images,
)
from src.utils.helpers import show_image, intersection_over_union, non_maximal_suppression
from src.utils.visualize import visualize_images_with_boxes_and_detections


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH / "model.pth")
    model.to(device)
    model.eval()

    # Initialize the scales that we will use to resize the image
    # SCALES = [1, 0.5, 0.3]
    SCALES = [0.5, 0.3]

    # Load the validation images
    validation_images = np.load(VALIDATION_NUMPY_PATH)

    # Initialize the detections, scores and file names
    detections = None
    scores = np.array([])
    file_names = np.array([])

    for i, image in enumerate(validation_images):
        start_time = timeit.default_timer()
        print(f"Processing image {i}/{len(validation_images)}...")
        image_scores = []
        image_detections = []
        image_name = str(i + 1).zfill(4) + ".jpg"

        for scale in SCALES:
            if scale * IMAGE_HEIGHT < DIM_HOG_WINDOW:
                break

            # Resize the image
            resized_image = resize(image, [IMAGE_HEIGHT * scale, IMAGE_WIDTH * scale])

            NUM_COLS = resized_image.shape[1] - DIM_HOG_WINDOW - 1
            NUM_ROWS = resized_image.shape[0] - DIM_HOG_WINDOW - 1

            for y in range(0, NUM_ROWS, 2):
                for x in range(0, NUM_COLS, 2):
                    resized_patch = resized_image[y : y + DIM_HOG_WINDOW, x : x + DIM_HOG_WINDOW]
                    window_tensor = transforms.ToTensor()(resized_patch).unsqueeze(0).to(device)
                    window_tensor = window_tensor.to(torch.float32)
                    with torch.no_grad():
                        pred = model(window_tensor)
                        score = pred[0][0].item()

                    if score > THRESHOLD:
                        x_min = int(x / scale)
                        y_min = int(y / scale)
                        x_max = int((x + DIM_HOG_WINDOW) / scale)
                        y_max = int((y + DIM_HOG_WINDOW) / scale)
                        image_detections.append([x_min, y_min, x_max, y_max])
                        image_scores.append(score)

                        # print("{:.10f}".format(score))
                        # show_image(resized_patch)
                        # cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        # show_image(image)

        if len(image_scores) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(image_detections), np.array(image_scores))
        if len(image_scores) > 0:
            if detections is None:
                detections = image_detections
            else:
                detections = np.concatenate((detections, image_detections))
            scores = np.append(scores, image_scores)
            file_names = np.append(file_names, np.repeat(image_name, len(image_scores)))

        end_time = timeit.default_timer()
        print(f"Process time for {i}/{len(validation_images)} was {end_time - start_time} seconds.")

    np.save(SOLUTION_DETECTIONS_PATH, detections)
    np.save(SOLUTION_SCORES_PATH, scores)
    np.save(SOLUTION_FILE_NAMES_PATH, file_names)


if __name__ == "__main__":
    # Collapse the images and annotations, save numpys
    # save_train_images_numpy()
    # save_validation_images_numpy()
    # collapse()

    # extract_cnn_images()

    run()

    # visualize_images_with_boxes_and_detections()
