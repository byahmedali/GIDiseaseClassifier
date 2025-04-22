# Gastrointestinal Diseases Classifier

A deep learning-based web application for classifying endoscopic images of the gastrointestinal tract into three conditions: Esophagitis, Polyps, and Ulcerative Colitis.

[GI Diseases Classifier](https://gidiseaseclassifier.streamlit.app/)

## Overview

This project uses a fine-tuned InceptionResNetV2 deep learning model to classify endoscopic images of the gastrointestinal tract. The model has been trained to identify three common GI conditions with high accuracy, making it a valuable tool for medical professionals and researchers.

### Key Features

- **Multi-class Classification**: Identifies three different gastrointestinal conditions
- **High Accuracy**: Achieves 98% accuracy on unseen endoscopic images
- **User-friendly Interface**: Simple web-based interface for easy image upload and classification
- **Real-time Results**: Instant classification with confidence scores for each condition
- **Automatic Model Loading**: Downloads the pre-trained model automatically on first run

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gi-diseases-classifier.git
   cd gi-diseases-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. On first run, the application will automatically download the pre-trained model

4. Upload an endoscopic image using the file uploader

5. View the classification results, including:
   - Predicted condition
   - Confidence scores for each condition

## Model Architecture

The application uses a fine-tuned InceptionResNetV2 model, which combines the Inception architecture with residual connections. This architecture allows for deeper networks with better feature extraction capabilities, which is crucial for detecting subtle differences in medical imaging.

### Preprocessing Pipeline

1. Resize the image to 299x299 pixels (required input size for InceptionResNetV2)
2. Convert the image to RGB format
3. Convert to NumPy array and add batch dimension
4. Convert to TensorFlow tensor for prediction

## Project Structure

```
gi-diseases-classifier/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Acknowledgements

We have acquired the test images from the [Kvasir dataset](https://datasets.simula.no/kvasir/). The dataset is described in:

Pogorelov, K., Randel, K. R., Griwodz, C., Eskeland, S. L., de Lange, T., Johansen, D., ... & Halvorsen, P. (2017). KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection. In *Proceedings of the 8th ACM on Multimedia Systems Conference (MMSys'17)* (pp. 164-169). ACM. https://doi.org/10.1145/3083187.3083212