# ECG Atrial Fibrillation Detection

This repository contains a machine learning pipeline for detecting atrial fibrillation (AFib) from ECG signals. The project demonstrates the end-to-end process from data preprocessing to model evaluation.

## Overview

Atrial fibrillation is a prevalent cardiac arrhythmia that can lead to serious health complications. This project aims to develop a robust model for identifying AFib from ECG data using various signal processing and machine learning techniques.

## Dataset

The dataset used in this project, `ECG_training.csv`, contains ECG signals along with their corresponding labels for training and evaluation purposes. Due to data size and security considerations, the dataset is not included directly in this repository.

To request access to the dataset, please send an email to [sanjaymythili2002@gmail.com](mailto:sanjaymythili2002@gmail.com). Include the following information in your request:

- Your full name
- Affiliation or organization
- Purpose of use

Upon verification, the dataset will be provided to you via email.

## Repository Structure

- **`ECG_Project.ipynb`**: Jupyter Notebook that includes data preprocessing, feature extraction, and model training.

## Installation

To run the project, ensure you have the following dependencies installed:

- **Python** (>=3.7)
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Tqdm**
- **PyWavelets**
- **Scikit-learn**

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn tqdm pywt scikit-learn
```

## Usage

1. **Data Preprocessing**: Load and normalize ECG signals from the provided dataset.
2. **Feature Extraction**: Extract features from the ECG signals, including time-domain, frequency-domain, and wavelet transform features.
3. **Model Training**: Train a Random Forest Classifier using the extracted features and evaluate its performance.
4. **Evaluation**: Assess the model using classification metrics and visualize the results with confusion matrices and classification reports.

## Results

The performance of the model is evaluated through a confusion matrix and a classification report. Results and metrics are visualized to assess the effectiveness of the model in detecting atrial fibrillation.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or additional features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyWavelets](https://pywavelets.readthedocs.io/) for wavelet transform functionalities.
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.

For more information, please refer to the [Jupyter Notebook](ECG_Project.ipynb) in this repository.
