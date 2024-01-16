# Computer Vision - Project 2 - The Flinstones Facial Detection & Recognition

## Author

- Hutan Mihai Alexandru
- Github: [hutanmihai](https://github.com/hutanmihai)
- LinkedIn: [Mihai-Alexandru Hutan](https://www.linkedin.com/in/hutanmihai/)

## The problem statement can be found here: [Problem.pdf](./Problem.pdf)

## For a detailed explanation of the solution please read the [report](./documentation.pdf).

## Required Libraries

python=3.11.5
numpy==1.26.3
opencv-python==4.9.0.80
matplotlib==3.8.2
jupyter==1.0.0
torch==2.1.2
torchvision==0.16.2

### Install required libraries using conda and pip

This is the recommended way of installing the required libraries.
You can also not use conda and install the libraries using pip, but you will have to make sure that you have the correct
version of python installed (3.11.5).

```bash
conda create --name computer-vision-project-2 python=3.11.5
conda activate computer-vision-project-2
pip install numpy==1.26.3 opencv-python==4.9.0.80 matplotlib==3.8.2 

# If you have a CUDA enabled GPU (Windows)

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# If you have a CUDA enabled GPU (Linux) / MacOS (CPU) / Windows (CPU)

pip install torch==2.1.2 torchvision==0.16.2

# If you want to train the networks, otherwise you can skip this step and they will be loaded from the models/ folder.

pip install jupyter==1.0.0

```

## How to run the project

Make sure you are in the root directory of the project.

- Windows - Powershell:

```bash
$env:PYTHONPATH='.' 
```

- Windows - CMD:

```bash
set PYTHONPATH=.
```

- Linux / MacOS:

```bash
export PYTHONPATH=.
```

### 1. (Optional) Train the networks

- If you want to train the networks, you first need to run `collapse()` and `extract_train_and_validation_patches()`.
  This will create the training and validation patches.
- `Note:` These will take somewhere between 30 seconds to 4 minutes depending on your machine.

```bash
python src/pre_training.py
```

- Then you can run the jupyter notebooks `task1_cnn.ipynb` and `task2_cnn.ipynb` to train the networks.
- You might need to change the paths as they are relative to the root of the project not the notebooks.
- `Note:` Task1 took me 30 minutes to train on a RTX3070 and Task2 took me about 5 minutes to train on the same GPU. If
  you don't have a powerful GPU or none at all, it will take a lot longer.
- The models will overwrite the ones in the models/ folder, therefore your results might be different from the ones in
  the report.

### 2. Run the project

- `Note:` Task1 took me about 2 hours to run on a RTX3070 and an I5-14600KF. If you don't have a powerful GPU or none at
  all, it will take a lot longer.
- Task2 runs really fast, because we use the results from task1, so be sure to run task1 first.

To run task1 on the test images run the following command, but if you want to run it on the validation images, just
remove the test flag.

```bash
python src/task1.py --test
```

For task2 run the following command

```bash
python src/task2.py
```

### 3. (Optional) Run the tests

```bash
python src/eval.py
```

### 4. (Optional) Visualize the results

```bash
python src/visualize_results.py
```
