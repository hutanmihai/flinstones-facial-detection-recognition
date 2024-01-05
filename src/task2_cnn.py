import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import transforms

from src.constants import (
    SOLUTION_DETECTIONS_PATH,
    SOLUTION_SCORES_PATH,
    SOLUTION_FILE_NAMES_PATH,
    VALIDATION_IMAGES_PATH,
    MODEL_PATH,
    COLOR_CHARACTER_MAPPING,
    SOLUTION_TASK2_PATH,
)
from src.utils.helpers import show_image

LABELS_MAP = {"barney": 0, "fred": 1, "wilma": 2, "betty": 3, "unknown": 4}

SOLUTION_DETECTIONS_BARNEY_PATH = SOLUTION_TASK2_PATH / "detections_barney.npy"
SOLUTION_SCORES_BARNEY_PATH = SOLUTION_TASK2_PATH / "scores_barney.npy"
SOLUTION_FILE_NAMES_BARNEY_PATH = SOLUTION_TASK2_PATH / "file_names_barney.npy"

SOLUTION_DETECTIONS_BETTY_PATH = SOLUTION_TASK2_PATH / "detections_betty.npy"
SOLUTION_SCORES_BETTY_PATH = SOLUTION_TASK2_PATH / "scores_betty.npy"
SOLUTION_FILE_NAMES_BETTY_PATH = SOLUTION_TASK2_PATH / "file_names_betty.npy"

SOLUTION_DETECTIONS_FRED_PATH = SOLUTION_TASK2_PATH / "detections_fred.npy"
SOLUTION_SCORES_FRED_PATH = SOLUTION_TASK2_PATH / "scores_fred.npy"
SOLUTION_FILE_NAMES_FRED_PATH = SOLUTION_TASK2_PATH / "file_names_fred.npy"

SOLUTION_DETECTIONS_WILMA_PATH = SOLUTION_TASK2_PATH / "detections_wilma.npy"
SOLUTION_SCORES_WILMA_PATH = SOLUTION_TASK2_PATH / "scores_wilma.npy"
SOLUTION_FILE_NAMES_WILMA_PATH = SOLUTION_TASK2_PATH / "file_names_wilma.npy"


def run_task2_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH / "really_good_result_task2_cnn.pth")
    model.to(device)
    model.eval()

    detections = np.load(SOLUTION_DETECTIONS_PATH)
    file_names = np.load(SOLUTION_FILE_NAMES_PATH)

    solutions = {
        "barney": [np.array([]), np.array([]), np.array([])],
        "betty": [np.array([]), np.array([]), np.array([])],
        "fred": [np.array([]), np.array([]), np.array([])],
        "wilma": [np.array([]), np.array([]), np.array([])],
    }

    for file_name, detection in zip(file_names, detections):
        image = cv.imread(str(VALIDATION_IMAGES_PATH / file_name))
        cropped_box = cv.resize(image[detection[1] : detection[3], detection[0] : detection[2]], (40, 40))
        cropped_box = cv.cvtColor(cropped_box, cv.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(cropped_box).unsqueeze(0).to(device)
        output = model(tensor)
        predicted = torch.argmax(output).item()
        score = output[0][predicted].item()
        character = list(LABELS_MAP.keys())[list(LABELS_MAP.values()).index(predicted)]

        print(f"Predicted: {character} with score: {score}")

        if character != "unknown":
            solutions[character][0] = np.append(solutions[character][0], np.array(detection))
            solutions[character][1] = np.append(solutions[character][1], score)
            solutions[character][2] = np.append(solutions[character][2], file_name)

    for character in solutions.keys():
        solutions[character][0] = solutions[character][0].reshape(-1, 4).astype(np.int32)
        if character == "unknown":
            continue
        if character == "barney":
            np.save(SOLUTION_DETECTIONS_BARNEY_PATH, solutions[character][0])
            np.save(SOLUTION_SCORES_BARNEY_PATH, solutions[character][1])
            np.save(SOLUTION_FILE_NAMES_BARNEY_PATH, solutions[character][2])
        elif character == "betty":
            np.save(SOLUTION_DETECTIONS_BETTY_PATH, solutions[character][0])
            np.save(SOLUTION_SCORES_BETTY_PATH, solutions[character][1])
            np.save(SOLUTION_FILE_NAMES_BETTY_PATH, solutions[character][2])
        elif character == "fred":
            np.save(SOLUTION_DETECTIONS_FRED_PATH, solutions[character][0])
            np.save(SOLUTION_SCORES_FRED_PATH, solutions[character][1])
            np.save(SOLUTION_FILE_NAMES_FRED_PATH, solutions[character][2])
        elif character == "wilma":
            np.save(SOLUTION_DETECTIONS_WILMA_PATH, solutions[character][0])
            np.save(SOLUTION_SCORES_WILMA_PATH, solutions[character][1])
            np.save(SOLUTION_FILE_NAMES_WILMA_PATH, solutions[character][2])

    # image_copy = image.copy()
    # cv.rectangle(image_copy, (detection[0], detection[1]), (detection[2], detection[3]), COLOR_CHARACTER_MAPPING[character], 2)
    # show_image(image_copy, title=file_name)


if __name__ == "__main__":
    run_task2_cnn()
