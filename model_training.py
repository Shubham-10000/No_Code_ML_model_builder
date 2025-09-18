import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

def model_training():
    st.title("Model Training & Evaluation")

    # Step 1: Load preprocessed data
    files = os.listdir('preprocessed_data') if os.path.exists('preprocessed_data') else []
    if not files:
        st.warning("No preprocessed data found. Please preprocess a file first.")
        return

    selected_file = st.selectbox("Select a preprocessed CSV file", files)
    df = pd.read_csv(os.path.join('preprocessed_data', selected_file))
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Step 2: Select target column
    target_column = st.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 3: Detect problem type
    if y.nunique() <= 20 and y.dtype in ['int64', 'int32']:
        problem_type = "classification"
        st.info("Detected as Classification Problem")
    else:
        problem_type = "regression"
        st.info("Detected as Regression Problem")

    # Step 4: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Define models
    if problem_type == "classification":
        models = {
            "Random Forest Classifier": RandomForestClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KNN Classifier": KNeighborsClassifier()
        }
    else:
        models = {
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "Linear Regression": LinearRegression()
        }

    # Step 6: Train, Evaluate, and Display Metrics
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            score = (accuracy_score(y_test, y_pred))*100
        else:
            score = (r2_score(y_test, y_pred))*100

        results.append((name, score, model, y_pred))

    # Step 7: Show model performance table
    results_df = pd.DataFrame(results, columns=["Model", "Score", "Trained_Model", "Predictions"])
    st.write("Model Performance:")
    st.dataframe(results_df[["Model", "Score"]])

    # Step 8: Best Model Selection
    best_model_row = results_df.loc[results_df["Score"].idxmax()]
    st.success(f"Best Model: {best_model_row['Model']} with Score: {best_model_row['Score']:.4f}")
    #SHOW NAME OF BEST MODEL
    st.write("Best Model Details:", best_model_row)
    best_model = best_model_row["Trained_Model"]
    
    
    # Step 9: Detailed Metrics for Best Model
    st.subheader("Detailed Metrics for Best Model")
    best_y_pred = best_model_row["Predictions"]

    if problem_type == "classification":
        st.write("**Accuracy Score:**", accuracy_score(y_test, best_y_pred))
        st.text("**Classification Report:**")
        st.text(classification_report(y_test, best_y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, best_y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.write("**Mean Squared Error:**", np.sqrt(mean_squared_error(y_test, best_y_pred))*100, "%")
        st.write("**R² Score:**", r2_score(y_test, best_y_pred))

    # Step 10: Save & Download Best Model
    if st.button("Save Best Model"):
        st.write("Saving the best model...")
        st.write(f"Model Name: {best_model_row['Model']}")
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{best_model_row['Model'].replace(' ', '_')}.pkl"
        joblib.dump(best_model_row["Trained_Model"], model_path)
        st.download_button(
            label="Download Model",
            data=open(model_path, "rb").read(),
            file_name=os.path.basename(model_path)
        )
