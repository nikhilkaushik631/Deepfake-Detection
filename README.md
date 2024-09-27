# Development of Deepfakes Detection Model Using Deep Learning Framework

## Overview

This project is designed to detect fake videos by analyzing and classifying video frames. The solution utilizes a pre-trained **ResNeXt50_32x4d** model combined with an **LSTM** network to process video sequences. The system extracts frames, detects faces, and classifies videos as either **REAL** or **FAKE**. It is optimized to run on **Google Colab**, leveraging its GPU capabilities.

## Features

- **Video Processing**: Extracts frames from videos and processes them for face detection.
- **Face Detection**: Utilizes the **MediaPipe** library to detect faces within video frames.
- **Deep Learning Model**: Employs a ResNeXt50_32x4d model and an LSTM network to classify video sequences.
- **Model Training and Evaluation**: Includes functionality to train the model, validate its performance, and generate confusion matrices.
- **Inference and Prediction**: Allows for prediction on new videos with confidence scoring and visualizations.
- **Data Augmentation**: Enhances the dataset by applying various augmentation techniques to the real video data to improve model robustness.
- **Face Extraction**: Utilizes a face extraction code to isolate faces from videos, creating a dataset specifically for training the model.

## Model Overview

### ResNeXt50_32x4d

**ResNeXt** is a powerful convolutional neural network architecture that improves upon the ResNet design. The key features include:

- **Cardinality**: Introduces the concept of cardinality (the number of paths in the architecture) in addition to depth and width, allowing for increased model capacity without substantially increasing computational cost.
- **Modular Design**: Built using a simple building block that allows for easy adjustments to the model's complexity.
- **Performance**: Achieves excellent performance on various image classification tasks, making it suitable for feature extraction in video frames.

**ResNeXt50_32x4d** specifically denotes a 50-layer network with 32 groups in its convolutions, making it highly effective for extracting rich features from input images.

### LSTM (Long Short-Term Memory)

**LSTMs** are a type of recurrent neural network (RNN) designed to handle sequential data, such as video frames. Their main advantages include:

- **Memory Cell**: Capable of learning long-term dependencies, allowing the model to remember information from earlier time steps while processing new data.
- **Gated Mechanisms**: Use forget, input, and output gates to control the flow of information, helping to mitigate issues like vanishing gradients that occur in standard RNNs.

In this project, the LSTM network is used to analyze the temporal sequences of the extracted frames, enabling the model to recognize patterns and classify video sequences effectively.

### PyTorch Framework

**PyTorch** is an open-source machine learning library widely used for deep learning applications. Key benefits include:

- **Dynamic Computation Graphs**: Allows for flexible model building and debugging, as the computation graph is created on-the-fly during runtime.
- **Rich Ecosystem**: Offers extensive libraries and tools, including `torchvision` for image processing and `torchtext` for natural language processing.
- **GPU Acceleration**: Facilitates seamless GPU usage for accelerated training and inference, making it ideal for computationally intensive tasks like deep learning.

This project leverages PyTorch to implement the ResNeXt and LSTM components, providing a robust platform for training and evaluating the deepfake detection model.

## Requirements

- **Python Version**: 3.x
- **Dependencies**: List of required libraries can be found in `requirements.txt`.

## Setup

### Install Dependencies

Create a virtual environment (optional) and install the required libraries using:

```bash
pip install -r requirements.txt
```

### Prepare Data

1. **Video Files**: Place your video files in the specified directory.
2. **Face Detection**: The `process_videos` function extracts and saves frames containing faces using MediaPipe.
3. **Model Weights**: Download and place the pre-trained model weights in the specified directory if necessary.
4. **Update Paths**: Modify the paths in the code for video files, face detection, and model checkpoints.

### Data Augmentation

- The data augmentation code processes real videos to balance the dataset. It applies techniques such as horizontal flip, random rotation, brightness adjustment, contrast adjustment, adding Gaussian noise, and applying Gaussian blur. This is essential for improving model performance and generalization.

### Face Extraction

- The face extraction code utilizes **MediaPipe** for face detection, isolating faces from video frames. It saves the detected faces as individual video files, ensuring that the model is trained on relevant data, focusing specifically on face features.

## Usage

### Data Preparation

1. Process videos to extract frames and detect faces using the `process_videos` function.
2. Use the `create_face_videos` function to save face-only videos.

### Label CSV File

Ensure you have a CSV file with labels corresponding to your videos. The CSV should have the following format:

```csv
file,label
video1.mp4,REAL
video2.mp4,FAKE
```

Upload this CSV file to your Google Drive and ensure it's in an accessible location, e.g., `/content/drive/MyDrive/file_names.csv`.

### Model Training

1. Set hyperparameters such as learning rate and number of epochs.
2. Train the model using the `train_epoch` function and validate it using the `test` function.
3. Save the trained model checkpoint.

### Inference

Load the trained model and use the `predict` function to classify new videos. Visualize the results using heatmaps and saved images.

### Evaluation

Generate plots for training and validation loss and accuracy. Review confusion matrices to assess model performance.

## Code Structure

- **Data Preparation**: Handles video frame extraction and face detection using MediaPipe.
- **Data Augmentation**: Applies various techniques to enhance the training dataset.
- **Face Extraction**: Isolates faces from videos to create a focused dataset for training.
- **Model**: Defines the hybrid ResNeXt50_32x4d and LSTM network.
- **Training and Evaluation**: Functions for training, validating, and visualizing model performance.
- **Inference**: Implements prediction and result visualization.

## Troubleshooting

- **Ensure Paths**: Verify that all file paths and directories are correctly set in the code.
- **Model Weights**: Confirm that the model weights are correctly placed and loaded.
- **Dependency Issues**: Check for any library version conflicts and ensure all required libraries are installed.
