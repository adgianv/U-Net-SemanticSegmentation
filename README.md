# Image Semantic Segmentation using U-Net

## Project Overview

This project focuses on image semantic segmentation using the U-Net architecture implemented with TensorFlow. The goal is to segment different objects within an image accurately. U-Net, a convolutional neural network, is particularly well-suited for biomedical image segmentation but has proven effective across various domains. This project leverages data augmentation to enhance the model's performance, achieving an F1 score of 0.71.

## Introduction

Semantic segmentation is a critical task in computer vision that involves labeling each pixel in an image with a corresponding class. This project implements a U-Net model using TensorFlow to perform semantic segmentation. The U-Net model is chosen for its effectiveness in segmenting images with high accuracy due to its unique architecture that captures both low-level and high-level features.

## Dataset

The dataset used in this project consists of annotated images suitable for semantic segmentation tasks. Each image has a corresponding ground truth mask. The dataset was split into training, validation, and test sets to evaluate the model's performance comprehensively.

## Model Architecture

The U-Net architecture is composed of an encoder-decoder structure:
- **Encoder**: A series of convolutional layers with down-sampling operations to capture context and features at multiple levels.
- **Decoder**: A series of up-sampling (using transpose convolution) layers that combine features from the encoder to produce a segmentation map.

The architecture ensures that spatial information is preserved while enabling deep feature extraction.

## Data Augmentation

Data augmentation techniques are employed to enhance the diversity of the training data, preventing overfitting and improving the model's generalization. The augmentation currently includes just horizontal flips but it can potentially include many other:
- Random rotations
- Horizontal and vertical flips
- Zoom and shift transformations
- Brightness and contrast adjustments

These augmentations are applied using an `ImageDataGenerator` from TensorFlow, ensuring real-time data augmentation during training.

## Training

The model is trained using the following configurations:
- **Loss Function**: Binary Focal Crossentropy
- **Optimizer**: Adam
- **Metrics**: F1 Score, IoU score, Accuracy, Precision, Recall

Training is performed for a fixed number of epochs with early stopping based on validation loss to prevent overfitting.

## Results

The model achieved an F1 score of 0.71 on the test set after data augmentation, indicating a good balance between precision and recall. The results demonstrate the effectiveness of the U-Net architecture and data augmentation techniques in semantic segmentation tasks.

## Usage

To use this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adgianv/U-Net-SemanticSegmentation.git
   cd U-Net-SemanticSegmentation

