import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load dataset (replace with your preprocessed file)
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\rabad\OneDrive\Desktop\New folder\ready\realistic_salary_prediction_dataset.csv")  # Replace with your path

# -----------------------------
# App Layout
st.set_page_config(layout="wide")
st.title("💼 Salary Prediction Dashboard")
st.markdown("This app allows you to evaluate multiple ML models to predict employee salary.")

# Load data
df = load_data()

if 'salary' not in df.columns:
    st.error("Target column 'salary' not found.")
    st.stop()

# -----------------------------
# Correlation Heatmap
st.subheader("📊 Feature Correlation Heatmap")
corr_matrix = df.select_dtypes(include=[np.number]).corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)

# -----------------------------
# Feature-Target split
X = pd.get_dummies(df.drop('salary', axis=1))
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Model dictionary
model_dict = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0),
    "ElasticNet Regression": ElasticNet(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, objective='reg:squarederror'),
    "Support Vector Regression": SVR(),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5)
}

top_models = {
    "Gradient Boosting": model_dict["Gradient Boosting"],
    "XGBoost": model_dict["XGBoost"]
}

# -----------------------------
# Sidebar Options
st.sidebar.header("⚙️ Model Options")
mode = st.sidebar.radio("Select Mode", ["Try All Models", "Use Top 2 Models Only"])
optional_override = st.sidebar.selectbox("Or Select Specific Model", ["None"] + list(model_dict.keys()))
run = st.sidebar.button("🔍 Run Prediction")

# -----------------------------
# Run Models
if run:
    st.info("Training models...")

    # Determine selected models
    if optional_override != "None":
        selected_models = {optional_override: model_dict[optional_override]}
    else:
        selected_models = model_dict if mode == "Try All Models" else top_models

    results = []
    predictions_df_list = []
    feature_importance_data = {}

    for name, model in selected_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "R²": round(r2, 4)
        })

        predictions_df_list.append(pd.DataFrame({
            "Model": name,
            "Actual Salary": y_test.values,
            "Predicted Salary": y_pred
        }))

        if hasattr(model, 'feature_importances_'):
            feature_importance_data[name] = pd.Series(model.feature_importances_, index=X.columns)

    # -----------------------------
    # Display Metrics
    st.subheader("📊 Model Performance")
    result_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    st.dataframe(result_df, use_container_width=True)

    # Custom bar chart for model comparison
    st.subheader("📉 Comparison of MAE, MSE, R²")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    result_df.set_index("Model")[["MAE", "MSE", "R²"]].plot(kind='bar', ax=ax_bar, colormap="viridis")
    ax_bar.set_title("Model Metrics Comparison")
    st.pyplot(fig_bar)

    best_model = result_df.iloc[0]["Model"]
    st.success(f"🏆 Best Model: {best_model}")

    # -----------------------------
    # Scatter Plot - Actual vs Predicted
    st.subheader(f"📈 Actual vs Predicted Salaries: {best_model}")
    best_pred_df = [df for df in predictions_df_list if df['Model'].iloc[0] == best_model][0]
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Actual Salary", y="Predicted Salary", data=best_pred_df, ax=ax1, color='dodgerblue')
    ax1.set_title(f"Actual vs Predicted - {best_model}")
    st.pyplot(fig1)

    # -----------------------------
    # Feature Importances
    if best_model in feature_importance_data:
        st.subheader(f"📌 Feature Importance - {best_model}")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fi = feature_importance_data[best_model].sort_values(ascending=False)[:15]
        sns.barplot(x=fi.values, y=fi.index, palette='mako', ax=ax2)
        ax2.set_title(f"Top Features Influencing Salary - {best_model}")
        st.pyplot(fig2)

    # -----------------------------
    # Predictions Table and Download
    st.subheader("📋 Final Predictions Sample")
    all_preds = pd.concat(predictions_df_list)
    st.dataframe(all_preds.head(10))

    csv = all_preds.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Predictions", data=csv, file_name="salary_predictions.csv", mime='text/csv')
