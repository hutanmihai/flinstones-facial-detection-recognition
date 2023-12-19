from pathlib import Path
from glob import glob

# Main paths
TRAIN_DATA_PATH = Path("../data/train")
VALIDATION_DATA_PATH = Path("../data/validation")
TEST_DATA_PATH = Path("../data/test")

# Images paths
BARNEY_IMAGES_PATH = TRAIN_DATA_PATH / "barney"
BETTY_IMAGES_PATH = TRAIN_DATA_PATH / "betty"
FRED_IMAGES_PATH = TRAIN_DATA_PATH / "fred"
WILMA_IMAGES_PATH = TRAIN_DATA_PATH / "wilma"
VALIDATION_IMAGES_PATH = VALIDATION_DATA_PATH / "images"

# Images globs
BARNEY_IMAGES = glob(str(BARNEY_IMAGES_PATH / "*.jpg"))
BETTY_IMAGES = glob(str(BETTY_IMAGES_PATH / "*.jpg"))
FRED_IMAGES = glob(str(FRED_IMAGES_PATH / "*.jpg"))
WILMA_IMAGES = glob(str(WILMA_IMAGES_PATH / "*.jpg"))

TRAIN_IMAGES = BARNEY_IMAGES + BETTY_IMAGES + FRED_IMAGES + WILMA_IMAGES
VALIDATION_IMAGES = glob(str(VALIDATION_IMAGES_PATH / "*.jpg"))

# Solution paths
TRAIN_SOLUTION_PATH = Path("../data/train_solution.txt")
TEST_SOLUTION_PATH = Path("../data/test_solution.txt")

# Ground truth paths
# TODO
