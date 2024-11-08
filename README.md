# MNIST Digit Recognition Using CNN

This project involves building a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is implemented using Python and TensorFlow/Keras.

## Project Overview

Handwritten digit recognition is a fundamental machine learning problem and is widely used as an introductory project for deep learning. In this project, we use a CNN to classify images from the MNIST dataset, which contains grayscale images of handwritten digits (0-9).

## Dataset

**Note**: The dataset is not included in this repository due to file size limitations. You can download the dataset directly from Kaggle:

- [MNIST Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer/data)

Download `train.csv` and `test.csv`, and place them in the project directory.

## Model Architecture

The CNN model is designed as follows:
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 1**: 2x2 pool size
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 2**: 2x2 pool size
- **Flatten Layer**: Converts 2D matrices to a 1D vector
- **Dense Layer**: 128 units, ReLU activation
- **Dropout Layer**: Dropout rate of 0.5 to prevent overfitting
- **Output Layer**: 10 units (for digits 0-9), softmax activation

## Code Structure

- `train.csv`: Training data with labels (to be downloaded from Kaggle)
- `test.csv`: Test data without labels (to be downloaded from Kaggle)
- `code.ipynb`: Python script for training the CNN model
- `submission.csv`: CSV file generated for submission (with predicted labels)

## Getting Started

### Prerequisites

Make sure you have Python installed along with the required libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`

Install dependencies using:
```bash
pip install numpy pandas matplotlib tensorflow
```
## Output
The script prints the training and validation accuracy after each epoch. The final submission.csv file contains the predicted labels for the test data.

## Results
The model achieves an accuracy of approximately 0.98800 on the Kaggle leaderboard. The accuracy can be further improved by tweaking the architecture or applying additional techniques like data augmentation.
