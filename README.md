# Speech Emotion Recognition (SER) using Custom CNN and MUA Module

## Overview
This project implements a deep learning pipeline for Speech Emotion Recognition (SER) using the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It extracts audio features and feeds them into a custom Convolutional Neural Network (CNN) architecture enhanced with a Multi-dimensional Unified Attention (MUA) module.

The model is specifically trained to classify four distinct emotional states:
* Neutral
* Angry
* Sad
* Happy

---

## Architecture Highlights
The core of this project is its custom deep learning model built with TensorFlow/Keras, which consists of two main components:

1.  **Multi-Level Feature (MLF) Extractor:** Uses three parallel 2D Convolutional pathways with varying kernel sizes (`4x4`, `2x8`, and `10x2`). This allows the network to capture multi-scale temporal and spectral features simultaneously.
2.  **Multi-dimensional Unified Attention (MUA) Module:** A custom attention mechanism that independently weights the feature maps across three dimensions:
    * **Time Dimension**
    * **Frequency Dimension**
    * **Channel Dimension** (using Self-Attention via Q, K, V convolutions)

---

## Data Processing
* **Feature Extraction:** Mel-frequency cepstral coefficients (MFCCs) are extracted from raw `.wav` files using the `librosa` library.
* **Standardization:** Extracted MFCCs are padded or truncated to a fixed shape of `(26, 300)` to ensure uniform input dimensions for the CNN.
* **Data Split:** The dataset is split into training and testing sets with a `60-40` ratio.

---

## Dependencies
Ensure you have the following Python libraries installed before running the notebook:

* `tensorflow`
* `librosa`
* `soundfile`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

---

## Usage Instructions

1.  **Set Up Dataset:** * Download the RAVDESS dataset.
    * Upload the dataset to your Google Drive.
    * Update the `ravdess_dir` and `save_directory` variables in the notebook to match the correct path in your Google Drive.
2.  **Run the Notebook:**
    * The first cell will prompt you to mount your Google Drive.
    * Execute the cells sequentially to load the audio files, extract MFCCs, build the model, and begin training.
3.  **Training details:**
    * The model is compiled using the `Adam` optimizer (learning rate = 0.0001) and `sparse_categorical_crossentropy` loss.
    * It is configured to train for 100 epochs with a batch size of 32.
4.  **Save/Load Model:**
    * After training, the model is automatically saved as `model.h5` in the specified directory.

---

## Evaluation
The notebook includes built-in cells to evaluate the model's performance on the unseen test set:
* Calculates and prints the final **Test Loss** and **Test Accuracy**.
* Generates a visual **Confusion Matrix** using `seaborn` to show the true vs. predicted classifications for the four emotion categories.
* Outputs a `pandas` DataFrame displaying a side-by-side comparison of the actual vs. predicted emotions for a quick manual review of the predictions.
