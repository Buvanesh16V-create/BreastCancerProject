# Breast Cancer Detection Using Machine Learning

## 1. Project Overview

This project develops a machine learning system to classify breast tumor measurements into malignant or benign categories using the Wisconsin Diagnostic Breast Cancer dataset.

The project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, model evaluation, and a Streamlit-based prediction interface.

This is an academic machine learning project and is not intended for clinical diagnosis.

## 2. Objectives

- Analyze the breast cancer dataset.
- Perform exploratory data analysis and visualization.
- Preprocess the data and split it into training, validation, and test sets.
- Train and tune Logistic Regression and Random Forest models.
- Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Pay particular attention to recall for the malignant class.
- Build an interactive prediction interface using Streamlit.

## 3. Dataset

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

The project uses the dataset available through scikit-learn's `load_breast_cancer()` function.

Dataset characteristics:
- 569 samples
- 30 numerical features
- 357 benign samples
- 212 malignant samples

Target labels used by scikit-learn:
- 0: Malignant
- 1: Benign

Features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, and other measurements.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- VS Code
- Google Colab

## 5. Methodology

### Step 1: Data Loading
Load the Wisconsin Diagnostic Breast Cancer dataset.

### Step 2: Exploratory Data Analysis
Examine class distribution, feature distributions, descriptive statistics, and correlations.

### Step 3: Data Preparation
Separate features and target labels. Use stratified splitting to create training, validation, and test sets.

### Step 4: Model Training
Train Logistic Regression and Random Forest models.

### Step 5: Hyperparameter Tuning
Use stratified cross-validation and grid search to tune model parameters, with special attention to malignant-class recall.

### Step 6: Model Evaluation
Evaluate the models using accuracy, malignant precision, malignant recall, malignant F1-score, ROC-AUC, and confusion matrices.

### Step 7: Threshold Selection
Choose a classification threshold using validation data and the F2-score for the malignant class.

### Step 8: Model Saving
Save the selected model, decision threshold, feature names, and model information using Joblib.

### Step 9: Web Application
Build an interactive Streamlit application to enter measurements and view predicted class probabilities.

## 6. Model Evaluation

The models are compared using:

- Accuracy
- Malignant-class precision
- Malignant-class recall
- Malignant-class F1-score
- ROC-AUC
- Confusion matrix

Insert the actual values from the project's test evaluation results here before submission.

Do not claim a performance score unless it has been measured on the held-out test set.

## 7. Running the Application

### Prerequisites

Install Python 3.11 and create a virtual environment.

Activate the environment in the project directory, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

The trained model must be available at:

models/best_model.joblib

## 8. Project Structure

```text
BreastCancerProject/
├── app.py
├── README.md
├── requirements.txt
├── models/
│   └── best_model.joblib
├── screenshots/
└── notebooks/
    └── breast_cancer_detection.ipynb
```

Add the notebook and screenshots when they are ready.

## 9. Limitations

- The application is for educational purposes only.
- The model is not clinically validated.
- Prediction quality depends on the dataset, preprocessing, model, and decision threshold.
- Default form values are demonstration values, not actual patient measurements.
- Model probabilities should not be interpreted as definitive medical risk.

## 10. Future Enhancements

- Add model explainability using SHAP or LIME.
- Improve input validation and interface accessibility.
- Deploy the application in a suitable cloud environment.
- Evaluate model generalization using additional independent datasets.

## 11. Conclusion

This project demonstrates a complete machine learning workflow for breast cancer classification, from exploratory data analysis and model tuning to evaluation and an interactive application.

The system provides an educational demonstration of how machine learning can classify numerical tumor measurements into two categories.