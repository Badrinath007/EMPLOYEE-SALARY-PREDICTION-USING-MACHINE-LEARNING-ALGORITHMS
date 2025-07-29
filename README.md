
# 💼 Employee Salary Prediction Using Machine Learning Algorithms

## 📌 Project Overview

This project aims to predict the **annual salaries of employees** based on a variety of input features using multiple regression-based machine learning models. The web interface is built using **Streamlit**, offering an interactive experience where users can choose between models, visualize performance, and understand feature influences.

---

## 📊 Problem Statement

With rising data availability in HR departments, it becomes increasingly beneficial to estimate employee salaries accurately for budgeting, recruitment, and planning. This project helps in:

- Understanding key salary-influencing features.
- Predicting salary using different ML regression models.
- Comparing model performance based on MAE, MSE, and R² scores.

---

## ✅ Features

- 📈 Train and compare **multiple regression models**.
- 🧠 Choose between **top 2 models** or **train all models**.
- 📊 Visualizations: actual vs. predicted plots, residual plots, and bar graphs.
- ⚙️ No dataset upload required — works on pre-cleaned dataset.
- 🎛️ User-friendly UI to select model options and view predictions.

---

## ⚙️ Algorithms Used

The following regression techniques are implemented:

- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor (SVR)

---

## 🖥️ Tech Stack

- **Frontend/UI**: Streamlit (Optional)
- **Backend**: Python
- **ML Libraries**: Scikit-learn, XGBoost, NumPy, Pandas (Use required libraries for your project)
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Badrinath007/EMPLOYEE-SALARY-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS.git
cd EMPLOYEE-SALARY-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS
```

### 2. Install Dependencies

Make sure you have Python 3.10+ installed. Then install the packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Application (Optional)

```bash
streamlit run main.py
```
---

## 📁 Project Structure

```
employee-salary-prediction/
│
├── main.py                                   # Streamlit UI code
├── realistic_salary_prediction_dataset.csv   # Cleaned dataset
├── README.md                                 # This file
└── requirements.txt                          # Python dependencies
```

---

## 📊 Sample Output

- Model Comparison Table
- Actual vs Predicted Salary Plot
- Residual Error Distribution
- Bar plot of model MAE, MSE, and R² scores

---

## 🧠 Model Evaluation Criteria

| Metric       | Description                                 |
|--------------|---------------------------------------------|
| MAE          | Mean Absolute Error                         |
| MSE          | Mean Squared Error                          |
| R² Score     | Coefficient of Determination (Model fit)    |

---

## 📌 Future Improvements

- Allow CSV upload for user datasets.
- Deploy the app using Heroku or Streamlit Cloud.

---
## Live Deployment
https://salarydashboard.streamlit.app/

---

## 🙋‍♂️ Author

**Badrinath A.**    
🔗 https://www.linkedin.com/in/badrinatha/

---

## 📝 License

This project is licensed under the **MIT License**.
