import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Streamlit App Title
st.title("Diabetes Prediction Model")
st.markdown("This app allows users to train and evaluate a diabetes prediction model.")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Data Exploration
    st.write("### Dataset Information")
    st.write(data.info())

    st.write("### Dataset Description")
    st.write(data.describe())

    # Plotting Feature Distributions
    st.write("### Feature Distributions")
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    # Splitting the dataset
    st.write("### Train-Test Split")
    test_size = st.slider("Select Test Size (in %):", 10, 50, 20) / 100
    random_state = st.number_input("Set Random State (default: 42):", value=42)

    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    st.write(f"Train Size: {len(X_train)} samples, Test Size: {len(X_test)} samples")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    st.write("### Model Training")
    n_estimators = st.slider("Select Number of Trees in the Forest:", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluation Metrics
    st.write("### Evaluation Metrics")
    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

    # Feature Importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importance)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV dataset to get started.")
