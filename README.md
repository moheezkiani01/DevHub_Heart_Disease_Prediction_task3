# Heart Disease Prediction

## 🌐 Live Deployment
Available soon

## 📊 Overview
This task builds binary classification models to predict whether a patient is at risk of heart disease based on various health metrics and medical features.

## 🎯 Objective
Build a model to predict whether a person is at risk of heart disease based on their health data.

## 📁 Dataset
**Heart Disease UCI Dataset**
- **Source:** Kaggle / UCI Machine Learning Repository
- **Size:** ~300 samples, 13+ features
- **Target:** Binary (0 = No disease, 1 = Disease present)
- **Features Include:**
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol Level
  - Fasting Blood Sugar
  - Resting ECG Results
  - Max Heart Rate Achieved
  - Exercise Induced Angina
  - ST Depression
  - Slope of Peak Exercise ST Segment
  - Number of Major Vessels
  - Thalassemia

## 📥 Dataset Download
**Required:** Download `heart.csv` from Kaggle

1. Visit: https://www.kaggle.com/datasets/ketanchandar/heart-disease-dataset
2. Download `heart.csv`
3. Place in: `Task3_Heart_Disease_Prediction/data/heart.csv`

Alternatively, use any Heart Disease UCI dataset from Kaggle.

## 🛠️ Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **matplotlib** - Plotting
- **seaborn** - Statistical visualizations
- **sklearn** - Machine learning models

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Clean dataset (handle missing values)
- ✅ Perform EDA to understand trends
- ✅ Train Logistic Regression model
- ✅ Train Decision Tree model
- ✅ Evaluate using accuracy, ROC curve, confusion matrix
- ✅ Highlight important features
- ✅ Compare model performance

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Download Dataset
Download `heart.csv` from Kaggle and place in `data/` folder

### 3. Run the Notebook
```bash
jupyter notebook Task3_Heart_Disease_Prediction.ipynb
```

## 📈 Key Outputs

### Models Implemented:
1. **Logistic Regression** - Linear classification
2. **Decision Tree Classifier** - Non-linear classification

### Evaluation Metrics:
- **Accuracy** - Overall correctness
- **Precision** - Correct positive predictions
- **Recall** - Detected disease cases
- **F1-Score** - Harmonic mean of precision/recall
- **ROC-AUC Score** - Model discriminative ability

### Visualizations:
1. Target distribution (pie chart & bar chart)
2. Correlation heatmap
3. Feature distributions (histograms)
4. Box plots by disease presence
5. Confusion matrices (both models)
6. ROC curves comparison
7. Feature importance charts
8. Model metrics comparison

### Expected Results:
- **Logistic Regression:**
  - Accuracy: ~85-88%
  - ROC-AUC: ~0.90-0.92

- **Decision Tree:**
  - Accuracy: ~80-85%
  - ROC-AUC: ~0.85-0.90

### Key Findings:
- ❤️ Chest pain type is a strong predictor
- 💓 Max heart rate achieved is important
- 📊 ST depression correlates with disease
- 🎯 Models achieve good predictive performance
- ⚠️ Logistic Regression slightly outperforms Decision Tree

## 💡 Skills Demonstrated
- Binary classification
- Medical data understanding and interpretation
- Model evaluation using ROC-AUC and confusion matrix
- Feature importance analysis
- Handling imbalanced datasets
- Clinical implications of ML models

## 📊 Notebook Structure
1. Import Libraries
2. Load Dataset
3. Data Inspection (missing values, types, shape)
4. Exploratory Data Analysis
   - Target distribution
   - Correlation analysis
   - Feature distributions
5. Data Preprocessing
   - Handle missing values
   - Feature-target separation
   - Train-test split
   - Feature scaling
6. Model Training
   - Logistic Regression
   - Decision Tree
7. Model Evaluation
8. Visualizations (confusion matrix, ROC curves)
9. Feature Importance
10. Model Comparison
11. Key Findings

## 🎓 Learning Outcomes
- Understand medical ML applications
- Build and evaluate classification models
- Interpret confusion matrices and ROC curves
- Analyze feature importance in healthcare
- Consider ethical implications of medical AI

## ⚠️ Important Notes
- **Medical Disclaimer:** This is an educational project only
- **Not Diagnostic:** Cannot replace medical professionals
- **Ethical Considerations:** Be careful with medical predictions
- **Data Privacy:** Handle medical data responsibly

## 🔗 Additional Resources
- [Heart Disease Dataset](https://www.kaggle.com/datasets/ketanchandar/heart-disease-dataset)
- [ROC Curve Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
- [Medical ML Ethics](https://www.nature.com/articles/s41591-021-01614-0)

---

**Status:** ✅ Complete  
**Estimated Time:** 60-75 minutes  
**Difficulty:** Intermediate
