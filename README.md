# Prodigy_DS_03.
Building a decision tree classifier for customer predictions.
# Task-03: Decision Tree Classifier for Predicting Customer Purchase

## Table of Contents
1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset Description](#dataset-description)
4. [Steps Involved](#steps-involved)
   - [1. Data Preprocessing](#1-data-preprocessing)
   - [2. Model Building](#2-model-building)
   - [3. Model Evaluation](#3-model-evaluation)
   - [4. Visualization](#4-visualization)
5. [Results](#results)
6. [Tools and Libraries Used](#tools-and-libraries-used)
7. [Future Work](#future-work)
8. [License](#license)

---

## Overview

This project demonstrates the use of a **decision tree classifier** to predict whether a customer will purchase a product or service based on demographic and behavioral data. The dataset utilized for this task is the **Bank Marketing dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

---

## Objective

The objective of this project is to predict whether a customer subscribed to a term deposit (`yes` or `no`) based on features such as age, job, marital status, education level, account balance, and past interactions with the bank during a marketing campaign.

---

## Dataset Description

The **Bank Marketing dataset** includes detailed customer data collected from direct marketing campaigns. Below are the key attributes:

### Demographic Data:
- `age`: Age of the customer.
- `job`: Type of job (e.g., admin, technician, etc.).
- `marital`: Marital status (e.g., married, single).
- `education`: Level of education (e.g., secondary, tertiary).

### Behavioral Data:
- `default`: Has credit in default (yes/no).
- `balance`: Average yearly account balance in euros.
- `housing`: Has a housing loan (yes/no).
- `loan`: Has a personal loan (yes/no).

### Campaign Data:
- `contact`: Type of communication used to contact the customer.
- `month`: Last contact month of the year.
- `duration`: Duration of the last contact in seconds.
- `campaign`: Number of contacts during this campaign.
- `pdays`: Days since last contact (from a previous campaign).
- `previous`: Number of contacts performed before this campaign.
- `poutcome`: Outcome of the previous marketing campaign.

### Target Variable:
- `y`: Whether the customer subscribed to a term deposit (yes/no).

### Source
- [Bank Marketing Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## Steps Involved

### 1. Data Preprocessing
- **Data Inspection**: Import and examine the dataset.
- **Handling Missing Data**: Fill missing values where necessary.
- **Encoding Categorical Variables**: Use one-hot encoding (`pd.get_dummies()`) for categorical data.
- **Splitting Data**: Divide the dataset into features (`X`) and target (`y`), then split into training and testing subsets using `train_test_split` from `sklearn.model_selection`.

### 2. Model Building
- Train a **Decision Tree Classifier** using the `sklearn.tree.DecisionTreeClassifier`.
- Apply splitting criteria such as **Gini Index** or **Entropy**.
- Tune hyperparameters like `max_depth` and `min_samples_leaf` to optimize the model and prevent overfitting.

### 3. Model Evaluation
- Evaluate model performance using:
  - **Accuracy Score**: Percentage of correct predictions.
  - **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.
  - **Classification Report**: Precision, recall, F1-score, and support for each class.
- Visualize the decision tree using `plot_tree()` for interpretability.

### 4. Visualization
- Use decision tree plots to understand the feature splits and thresholds applied in the model.

---

## Results

- The decision tree classifier achieved an accuracy of **88.72%** on the test set.

---

## Tools and Libraries Used

- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation and preprocessing.
  - `scikit-learn` for building and evaluating the decision tree.
  - `matplotlib` for visualizing the decision tree.
- **Environment**: Jupyter Notebook or any Python IDE.

---

## Future Work

- Experiment with other classification algorithms like Random Forests or Gradient Boosting.
- Perform feature engineering to improve model performance.
- Deploy the model using Flask or FastAPI for real-time predictions.
- Explore model interpretability using tools like SHAP or LIME.

---
