# Lung Disease Prediction

This project is aimed at predicting lung diseases using deep learning and X-ray images. The repository contains the code and model necessary to classify X-ray images for potential lung disease, making use of a trained neural network model.

## Features

- **Jupyter Notebook (`X_rays.ipynb`)**: Contains data exploration, preprocessing, model training, and evaluation steps.
- **Flask App (`app.py`)**: A web application that allows users to upload X-ray images and receive predictions from the trained model.
- **Pre-trained Model (`model.h5`)**: The saved neural network model used for inference.
- **Requirements (`requirements.txt`)**: List of Python dependencies needed to run the code and app.
- **Assets Folder (`assests/`)**: Contains additional resources or files needed by the app (e.g., images, web assets).

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nexux69/Lung-Disease-Prediction.git
   cd Lung-Disease-Prediction
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Running the Notebook

Open and run `X_rays.ipynb` in Jupyter Notebook to see the data analysis, model training, and evaluation process.

#### Running the Web App

1. Ensure the `model.h5` file is present in the root directory.
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5000` to access the web app and upload X-ray images for prediction.

## Project Structure

```
.
├── X_rays.ipynb         # Data analysis, training, evaluation
├── app.py               # Flask web app
├── assests/             # Assets/resources for the app
├── model.h5             # Trained deep learning model
└── requirements.txt     # Python dependencies
```

## Notes

- Make sure all dependencies are installed before running the app or notebook.
- The model was trained on a set of X-ray images. For best results, use similar image formats when making predictions.

## License

This project does not currently specify a license.

---

**Author:** [Nexux69](https://github.com/Nexux69)
