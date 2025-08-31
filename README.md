# Lung Disease Prediction using Deep Learning (DenseNet121 + Streamlit)

## Description

This project predicts lung diseases—**COVID-19, Normal, Pneumonia, and Tuberculosis**—using advanced AI/ML techniques. The core model is trained on a curated Kaggle dataset of chest X-ray images. The objective is to provide a demo web application where users can upload X-ray images and receive disease predictions in real time.  
The entire pipeline, from model development to deployment, is fully self-implemented by me (**Faiz Shaikh**) with no external licenses or proprietary code.

---

## Key Features

- **Dataset Source:** Kaggle chest X-ray dataset covering COVID-19, Pneumonia, Tuberculosis, and Normal cases.
- **Preprocessing:** Utilizes Keras `ImageDataGenerator` for robust image augmentation and normalization.
- **Transfer Learning:** Leverages **DenseNet121** pretrained on ImageNet for high accuracy and efficient feature extraction.
- **Custom Classifier:** Fine-tuned dense layers tailored for 4-class disease classification.
- **Training Environment:** Complete model development and training in Google Colab for scalability and reproducibility.
- **Model Artifact:** Trained model exported as `.h5` file for easy integration and deployment.
- **Evaluation Metrics:** Assessed via accuracy, loss curves, confusion matrix, and detailed classification report.
- **Deployment:** Interactive web app served via **Streamlit Cloud** for user-friendly, real-time inference.

---

## How it Works (Step by Step)

1. **Data Acquisition & Preparation**
   - Download chest X-ray dataset directly from Kaggle in Google Colab.
   - Split dataset into **training**, **validation**, and **test** subsets.

2. **Model Architecture**
   - Build model using **DenseNet121** with frozen convolutional base (transfer learning).
   - Add custom dense layers for final multi-class classification.

3. **Training**
   - Use `AdamW` optimizer for robust convergence.
   - Apply `EarlyStopping` callback to avoid overfitting.
   - Monitor validation accuracy and loss.

4. **Evaluation**
   - Achieve approximately **89% accuracy** on the test set.
   - Analyze performance with confusion matrix and classification report.

5. **Prediction & Deployment**
   - Test the trained model on unseen chest X-ray images.
   - Deploy the model as a Streamlit web application for instant predictions.

---

## Demo Link

[Click here to try the Lung Disease Prediction App](https://lung-disease-prediction-faiz-shaikh.streamlit.app/)

---

## Screenshots

![App Screenshot 1](https://github.com/user-attachments/assets/007125ca-1ed5-49b1-8d69-64934599ed1d)
![App Screenshot 2](https://github.com/user-attachments/assets/9dcc2845-1ad9-4cb4-a6cd-25be707e0bc4)
![App Screenshot](screenshot.png)

---

## Tech Stack

- **Python 3**
- **TensorFlow** / **Keras** (DenseNet121, ImageDataGenerator)
- **Pandas**, **NumPy**, **Matplotlib** (data manipulation & visualization)
- **Streamlit** (web UI & deployment)
- **Google Colab** (model training environment)

---

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nexux69/Lung-Disease-Prediction.git
   cd Lung-Disease-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```
   The app launches at `http://localhost:8501/`.

---

## Acknowledgement

- This project is **100% self-implemented** by **Faiz Shaikh**.
- No external license or proprietary code is used.

---

## Future Scope

- Integrate more advanced deep learning models (e.g., EfficientNet, Vision Transformers).
- Expand dataset for improved generalization and accuracy.
- Enhance UI/UX with richer visual feedback and patient history support.
- Add explainability (Grad-CAM, saliency maps) for AI predictions.

---

## Contributing

> Currently, external contributions are not being accepted.  
> For feedback or suggestions, please open an issue.

---
