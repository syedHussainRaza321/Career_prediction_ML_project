import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- Page setup ---
st.set_page_config(page_title="Career Prediction App", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=80)
st.sidebar.title("Career Prediction")
page = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Model & Prediction", "Conclusion"])

# --- Generate model and preprocessing files if not exist ---
if not os.path.exists("rf_model.pkl"):
    df = pd.read_csv("Data_final.csv")

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(df["Career"])
    joblib.dump({label: idx for idx, label in enumerate(le.classes_)}, "label_mapping.pkl")

    # Preprocessing
    X = df.drop("Career", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    # Train model
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "rf_model.pkl")

# --- Load model and mappings ---
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_mapping = joblib.load("label_mapping.pkl")
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# --- Introduction Page ---
if page == "Introduction":
    st.title("🎯 Welcome to Career Prediction App")
    st.image("https://img.freepik.com/free-vector/job-interview-concept-illustration_114360-4853.jpg", width=500)
    st.markdown("""
    This app helps you discover your ideal career path based on your personality traits (OCEAN model) and aptitude scores.

    ---
    ### 🧬 What is the OCEAN Personality Test?
    The **OCEAN** model, also known as the Big Five personality traits, is widely used in psychology and career guidance. These traits influence decision-making, collaboration, creativity, and work style.

    - **Openness**: Curiosity, creativity, and openness to new experiences.
    - **Conscientiousness**: Organization, discipline, and goal-orientation.
    - **Extraversion**: Sociability, assertiveness, and enthusiasm.
    - **Agreeableness**: Compassion, cooperation, and trust in others.
    - **Neuroticism**: Tendency to experience emotional instability and stress.

    ### 🧠 Aptitude Skills
    Aptitude tests measure your natural abilities in areas like:
    - **Numerical**: Math and logic
    - **Spatial**: Visualization and manipulation of shapes
    - **Perceptual**: Attention to detail and identifying patterns
    - **Abstract**: Problem-solving using concepts and symbols
    - **Verbal**: Language understanding and communication

    Use the sidebar to explore the data, test the model, and see your career fit!
    """)

# --- EDA Page ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df = pd.read_csv("Data_final.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Career Distribution")
    fig1, ax1 = plt.subplots()
    df['Career'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Number of Students per Career")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Career")
    st.pyplot(fig1)

    st.subheader("Feature Distributions")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.multiselect("Select features to visualize", numeric_columns, default=numeric_columns[:3])
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax, color='orange')
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Model & Prediction Page ---
elif page == "Model & Prediction":
    st.title("🔍 Predict Your Career")
    st.image("https://img.freepik.com/free-vector/choose-career-concept-illustration_114360-5055.jpg", width=500)
    st.write("Adjust the sliders based on your personality and aptitude scores, then click **Predict Career**.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            O = st.slider("Openness", 1.0, 10.0, 5.0)
            C = st.slider("Conscientiousness", 1.0, 10.0, 5.0)
            E = st.slider("Extraversion", 1.0, 10.0, 5.0)
            A = st.slider("Agreeableness", 1.0, 10.0, 5.0)
            N = st.slider("Neuroticism", 1.0, 10.0, 5.0)
        with col2:
            num = st.slider("Numerical Aptitude", 1.0, 10.0, 5.0)
            spa = st.slider("Spatial Aptitude", 1.0, 10.0, 5.0)
            perc = st.slider("Perceptual Aptitude", 1.0, 10.0, 5.0)
            absr = st.slider("Abstract Reasoning", 1.0, 10.0, 5.0)
            verb = st.slider("Verbal Reasoning", 1.0, 10.0, 5.0)
        submit = st.form_submit_button("🎯 Predict Career")

    if submit:
        user_input = np.array([[O, C, E, A, N, num, spa, perc, absr, verb]])
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)[0]
        predicted_career = inv_label_mapping[prediction]

        st.markdown("""
        <div style='padding: 20px; background-color: #e6f7ff; border-radius: 10px; border: 1px solid #91d5ff;'>
        <h3 style='color: #0050b3;'>🎓 Your predicted career is:</h3>
        <h2 style='color: #1890ff;'><strong>{}</strong></h2>
        </div>
        """.format(predicted_career), unsafe_allow_html=True)

# --- Conclusion Page ---
elif page == "Conclusion":
    st.title("📝 Project Conclusion")
    st.image("https://img.freepik.com/free-vector/target-achievement-concept_23-2148423323.jpg", width=500)
    st.markdown("""
    ### Summary:
    This machine learning-powered app demonstrates how psychological (OCEAN) and aptitude traits can guide career predictions.

    - ✅ Clean interface with data-driven insights
    - 🧠 Random Forest used for accurate predictions
    - 🔍 Interactive user inputs for live predictions

    We hope this app helps you gain insights into your potential career path!

    **Thank you for visiting!** 💼
    """)
    st.balloons()
