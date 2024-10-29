# Helmet Detection Model

This repository contains the implementation of a Helmet Detection Model using TensorFlow and Keras. The model is designed to detect whether a person is wearing a helmet or not in images. The dataset used includes images annotated with bounding boxes and labels indicating whether a helmet is present.

## **Table of Contents**
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## **Introduction**

Helmet detection is crucial for ensuring safety standards are met in various environments, such as construction sites and roadways. This project leverages a Faster R-CNN approach with a ResNet50 backbone to accurately detect helmets in images.

## **Requirements**

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- lxml

Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib lxml


# Data Preparation

The dataset consists of images and their corresponding XML annotation files.

- **Annotations Directory:** `/content/drive/MyDrive/annotations`
- **Images Directory:** `/content/drive/MyDrive/images`

## Parsing XML Annotations

The annotations are in XML format. The `parse_annotation` function extracts object names and bounding boxes from the XML files.

```python
def parse_annotation(annotation_file):
    # Parsing logic


# Loading Data

The `load_data` function loads and preprocesses the images and annotations for training.

```python
def load_data(annotations_dir, images_dir):
    # Loading and preprocessing logic


# Model Architecture

The model is based on Faster R-CNN with a ResNet50 backbone, pre-trained on ImageNet.

```python
def create_faster_rcnn():
    # Model creation logic


# Training

The model is compiled and trained using the following parameters:

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# Evaluation

The model can be evaluated on a separate test set. Here, we demonstrate how to load a saved model and make predictions on new images.


## Preprocess Input Image

```python
def preprocess_image(image_path):
    # Preprocessing logic



## Interpret Predictions
```python
def interpret_prediction(prediction, threshold=0.6):
    # Interpretation logic


# Usage
To use the model for predicting helmet presence on new images, follow these steps:

## Load the saved model:
```python
loaded_model = tf.keras.models.load_model("/content/drive/MyDrive/helmet_detection_model.h5")

## Make Predictions:
```python
prediction = predict_image(image_path, loaded_model)
result = interpret_prediction(prediction[0][0])


## Display Image with Prediction:
```python
display_image_with_prediction(image_path, result)


# Result and Accuracy

The model is working fine with accuracy of 81% and is able to successfully detect the images With Helmets and Without Helmets.



