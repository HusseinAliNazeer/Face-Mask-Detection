# Face Mask Detection

This repository contains the code and materials for a face mask detection system using computer vision and deep learning techniques.

## Description

The face mask detection system is designed to identify whether an individual is wearing a face mask or not. The system uses a convolutional neural network (CNN) model to classify images as either "with mask" or "without mask". The model is trained on a dataset of images of people wearing and not wearing face masks.

## Files

The repository contains the following files:

- `train_mask_detector.py`: Python script to train the CNN model on the face mask dataset
- `detect_mask_image.py`: Python script to detect face masks in a single image
- `detect_mask_video.py`: Python script to detect face masks in a video stream
- `mask_detector.model`: Trained CNN model for face mask detection
- `face_detector`: Pre-trained face detector model for detecting faces in images and videos

## Requirements

To run the face mask detection system, you will need to have the following libraries installed:

- OpenCV
- numpy
- tensorflow
- keras

## Usage

To train the CNN model on the face mask dataset, run the following command:

```
python train_mask_detector.py --dataset dataset
```

To detect face masks in a single image, run the following command:

```
python detect_mask_image.py --image examples/example_01.png --face face_detector --model mask_detector.model
```

To detect face masks in a video stream, run the following command:

```
python detect_mask_video.py --face face_detector --model mask_detector.model
```
