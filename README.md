# Diabetes Prediction using KNN Classification

## Project Overview
This project uses the K-Nearest Neighbors algorithm to predict whether a person is likely to have diabetes based on health-related features such as glucose level, BMI, blood pressure, insulin, age, and other medical measurements.

The main goal of this project is to understand how KNN classification works and how feature scaling affects distance-based machine learning models.

## Dataset
The dataset used in this project is the Pima Indians Diabetes Dataset from Kaggle.

Target column:
- `Outcome`
  - `0` = Not Diabetic
  - `1` = Diabetic

## Features
The dataset contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Technologies Used
- Python
- Pandas
- Scikit-learn
- StandardScaler
- K-Nearest Neighbors
- Streamlit
- Joblib

## Project Workflow
1. Loaded the dataset
2. Separated input features and target column
3. Split the data into training and testing sets
4. Applied feature scaling using StandardScaler
5. Built a KNN classification model
6. Tested different K values
7. Selected the best K value
8. Evaluated the final model using accuracy, confusion matrix, and classification report
9. Built a simple Streamlit UI for prediction

## Model Performance
The KNN model was evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-score

The model achieved good performance for a beginner-level classification project after testing different K values.

## Streamlit App
A simple Streamlit web app was created where users can enter health-related values and get a prediction.

To run the app:

```bash
streamlit run app.py
