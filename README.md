# University Dropout Prediction

## Overview

This project aims to predict the likelihood of university students dropping out based on various factors, including academic performance, socio-economic status, and demographic data. The project leverages machine learning algorithms to create a predictive model that can assist in identifying at-risk students early, enabling interventions that could help retain them.

### Key Features:
- Data preprocessing to clean and prepare the dataset for machine learning models.
- Exploratory Data Analysis (EDA) to uncover insights about dropout patterns.
- Model training using several machine learning algorithms such as SVM, Random Forest, Gradient Boosting, KNN, and Logistic Regression.
- Hyperparameter optimization with RandomizedSearchCV for better model performance.
- Evaluation metrics including accuracy, precision, recall, and F1 score to assess model performance.
- Use of an ensemble model (Voting Classifier) for improved prediction accuracy.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Model Evaluation](#model-evaluation)
5. [Results and Discussion](#results-and-discussion)
6. [Conclusion](#conclusion)
7. [Installation and Usage](#installation-and-usage)

## Problem Statement

University dropout is a significant concern for educational institutions. This project aims to predict which students are at risk of dropping out, based on factors such as academic performance, age, socio-economic background, and more. By developing a predictive model, universities can take early action to provide support and improve student retention rates.

## Dataset

The dataset used for this project includes data on university students, including variables like academic performance, demographics, socio-economic status, and other relevant factors. The target variable is whether a student dropped out or not.

### Key Features:
- **Academic Performance**: Grades, curricular units authorized.
- **Demographic Information**: Age, gender, and socio-economic status.
- **Behavioral Data**: Participation in extracurricular activities, attendance, etc.

## Approach

1. **Data Preprocessing**: The dataset was cleaned and missing values were handled. Categorical variables were encoded, and numerical features were scaled for model training.
2. **Exploratory Data Analysis (EDA)**: Key insights were gathered regarding factors influencing student dropout. This involved visualizations like correlation matrices and dropout rate distributions.
3. **Model Training**: Several models were trained to predict dropout risk:
   - **Gradient Boosting**
   - **Random Forest**
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**
   - **K-Nearest Neighbors (KNN)**
4. **Hyperparameter Optimization**: RandomizedSearchCV was used to fine-tune each model’s hyperparameters to improve performance.
5. **Ensemble Method**: A Voting Classifier (soft voting) was employed to improve the model’s predictive power by combining multiple models.

## Model Evaluation

### Models Evaluated

The following models were configured and tested:
- **Gradient Boosting**: Optimized for the number of estimators, learning rate, and maximum depth.
- **Random Forest**: Tuned for the number of estimators, maximum depth, minimum samples split, and minimum samples leaf.
- **Logistic Regression**: Adjusted for regularization strength.
- **SVM**: Evaluated with a radial basis function kernel and varying regularization parameters.
- **KNN**: Assessed with different numbers of neighbors and weight options.

Each model was optimized using `RandomizedSearchCV` to identify the best hyperparameters based on cross-validation accuracy.

## Results and Discussion

### Results

The performance of various machine learning models was evaluated to identify the most effective approach for predicting university student dropout rates. The models assessed included Gradient Boosting, Random Forest, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

**Accuracy Scores of Models**

- **Support Vector Machine (SVM)**: SVM emerged as the best-performing model with the highest accuracy score of **0.8446**.
- **Gradient Boosting**: Achieved an accuracy score of **0.8432**.
- **Logistic Regression and KNN**: Both models were efficient but had lower accuracy compared to SVM and Gradient Boosting.
- **Random Forest**: Showed competitive performance with significant accuracy scores.

Despite longer optimization times, SVM’s superior accuracy indicates it is the most effective model for this classification task.

### Discussion

#### Exploratory Data Analysis (EDA)

- **Status Distribution**: 32.12% of students had not pursued further education.
- **Correlation Analysis**: Strong correlations were found between academic performance and student retention.

#### Ensemble Model Configuration

A **VotingClassifier** with a soft voting mechanism improved the overall forecast accuracy by combining multiple models. A confusion matrix and classification report provided insights into performance.

#### Model Evaluation Metrics

Precision, recall, and F1 score were computed for each model, revealing that SVM had the best overall metrics.

#### Optimization of Hyperparameters and Learning Curves

`RandomizedSearchCV` was used for hyperparameter tuning, and learning curves were plotted to identify potential issues with underfitting or overfitting.

## Conclusion

Based on the results, we can conclude that **Support Vector Machine (SVM)** achieved the highest accuracy score, outperforming other models. Key advantages of SVM include:
- **High-dimensional efficacy**
- **Resistance to overfitting**
- **Effective margin separation**
- **Versatility with different kernel functions**

The SVM model, followed by Gradient Boosting, provided the best performance for predicting university dropout rates.
