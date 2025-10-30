# Mobile Price Range Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

##  Project Overview

This project builds a **classification model** to predict the **price range of mobile phones** based on their technical specifications.  
It was developed as part of a **Data Science Minor Project**, showcasing end-to-end data preprocessing, model training, and evaluation.

---

##  Objective

To predict whether a mobile phone belongs to **Low**, **Medium**, **High**, or **Very High** price range using features like:
- Battery Power  
- RAM  
- Processor Speed  
- Camera Specs (Primary & Front)  
- Connectivity Features (3G, 4G, Wi-Fi, Bluetooth, Dual SIM)  
- Display Dimensions  

---

##  Dataset Details

- **Rows:** 2000  
- **Columns:** 21  
- **Target Variable:** `price_range` (0 = Low, 1 = Medium, 2 = High, 3 = Very High)  
- **Source:** Provided dataset during Data Science internship / coursework  

---

##  Steps Involved

### 1️⃣ Data Preprocessing
- Checked for **null values** and **duplicates**
- Cleaned and standardized data  
- Verified **target class distribution**

### 2️⃣ Data Splitting
- Train-Test Split: 75% training, 25% testing  
- Features (`X`) and Target (`y`) separated before modeling

### 3️⃣ Model Building and Evaluation
Applied and compared several ML algorithms:

| Model | Training Accuracy | Test Accuracy | Remarks |
|:------|:-----------------:|:--------------:|:--------|
| Logistic Regression | 64.5% | 61.6% | Baseline model |
| K-Nearest Neighbors (KNN) | 95.2% | 93.8% | Strong performance |
| SVM (Linear Kernel) | 98.3% | **96.4%** | Best accuracy |
| SVM (RBF Kernel) | 95.5% | 95.2% | Excellent generalization |
| Decision Tree | 100% | 80.4% | Overfitting observed |
| Random Forest | 100% | 88.0% | Balanced performance |

### 4️⃣ Performance Metrics
- **Confusion Matrix**
- **Precision**, **Recall**, **F1-Score**
- **Overall Accuracy**

---

##  Results Summary

-  **Best Performing Model:** SVM Classifier (Linear Kernel)
-  **Highest Test Accuracy:** **96.4%**
-  Models ranked by accuracy:

| Rank | Model | Accuracy |
|------|--------|----------|
| 1️⃣ | SVM (Linear) | 96.4% |
| 2️⃣ | SVM (RBF) | 95.2% |
| 3️⃣ | KNN | 93.8% |
| 4️⃣ | Random Forest | 88.0% |
| 5️⃣ | Decision Tree | 80.4% |
| 6️⃣ | Logistic Regression | 61.6% |

---

##  Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| ML Algorithms | Logistic Regression, KNN, SVM, Decision Tree, Random Forest |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook / Google Colab |

---
##  Key Learnings

Feature selection and normalization can greatly affect model performance.

SVM performed best for this dataset, showing its effectiveness in high-dimensional spaces.

Comparing multiple algorithms is essential for finding optimal models.
## Author

Ajitha Kumaravel
M.Sc. Physics | Data Science & Machine Learning Enthusiast
VIT Chennai
ajithakumaravel200@gmail.com



# Run the notebook
jupyter notebook MINOR_PROJECT_AJITHA.ipynb
