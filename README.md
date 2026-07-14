# 🏅 Optimizing University Sport Investment Using Popularity and Stress Analysis

## 📖 Project Overview

This project analyzes student sports participation, stress levels, and lifestyle factors to support evidence-based decisions for university sports investment.

Using the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology, the project explores how academic workload, sports participation, sleep, activity level, and gender influence student stress.

The application is developed using **Python** and **Streamlit**, providing an interactive dashboard for descriptive statistics, hypothesis testing, visualization, and machine learning.

---

## 🎯 Objectives

- Analyze student sports participation patterns.
- Identify the most popular sports for future university investment.
- Examine factors affecting student stress.
- Evaluate whether sports participation reduces stress.
- Predict student stress using Machine Learning.
- Provide data-driven recommendations for university decision-makers.

---

## ❓ Research Questions

1. Does academic workload significantly increase student stress?
2. Does sports participation reduce stress?
3. Do male and female students experience different stress levels?
4. Which sports should receive future university investment?

---

## 📊 Dataset

The dataset contains information collected from Siam University students, including:

- Gender
- Sport
- Academic Workload
- Sports Hours per Week
- Sleep Hours
- Activity Level
- Height
- Stress Level

After data cleaning:

- Removed unnecessary columns
- Renamed inconsistent column names
- Removed "Prefer Not To Say" responses
- Prepared numerical variables for statistical analysis and machine learning

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- SciPy

---

## 📈 Features

### Data Cleaning
- Remove missing and unnecessary data
- Prepare dataset for analysis

### Descriptive Statistics
- Mean
- Median
- Mode
- Standard Deviation
- IQR
- Summary Statistics

### Data Visualization

- Sport Popularity Bar Chart
- Academic Workload vs Stress
- Gender vs Stress
- Gender vs Sports Hours
- Correlation Heatmap
- Scatter Plots
- Box Plots

### Statistical Analysis

- One-Way ANOVA
- Independent Sample t-test
- Pearson Correlation

### Machine Learning

#### Linear Regression
- Stress prediction
- Model coefficients
- R²
- RMSE
- MSE

#### Decision Tree Classifier
- Stress classification
- Decision Tree visualization
- Accuracy
- Classification Report
- Feature Importance

---

## 📌 Key Findings

- Academic workload is the strongest predictor of student stress.
- Female students reported significantly higher stress levels than male students.
- Sports participation shows a weak negative relationship with stress.
- Badminton is the most popular sport.
- Basketball is the second most popular sport.
- Investing in Badminton and Basketball benefits approximately 69% of participating students.

---

## 💡 Recommendations

- Increase investment in Badminton facilities.
- Improve Basketball infrastructure.
- Develop stress reduction programs for students with high academic workload.
- Encourage greater sports participation across campus.
- Implement targeted wellbeing programs for female students.

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/your-username/siam-university-sports-analysis.git
```

### Install required packages

```bash
pip install streamlit pandas numpy matplotlib plotly scipy scikit-learn openpyxl
```

### Run the application

```bash
streamlit run data_analyze.py
```

---

## 📂 Project Structure

```
Sports-Analysis/
│
├── data_analyze.py
├── data_cleaning.py
├── Cleaned_Data.xlsx
├── README.md
└── requirements.txt
```

---

## 📊 CRISP-DM Framework

- Business Understanding
- Data Understanding
- Data Preparation
- Modeling
- Evaluation
- Deployment

---

---

## 🎓 Course

Data Science

Siam University

---

## 📜 License

This project is developed for educational purposes as part of a Data Science course at Siam University.
