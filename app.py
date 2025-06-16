import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- Page setup ---
st.set_page_config(page_title="Career Prediction App", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=80)
st.sidebar.title("Career Prediction")
st.sidebar.markdown("""---
Built with ❤️ using Streamlit
""")
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
    The **OCEAN** model, also known as the Big Five personality traits, is widely used in psychology and career guidance:

    - **Openness**: Curiosity, creativity, and imagination.
    - **Conscientiousness**: Responsibility and organizational skills.
    - **Extraversion**: Energy, assertiveness, and sociability.
    - **Agreeableness**: Compassion and cooperative nature.
    - **Neuroticism**: Sensitivity to stress and emotional fluctuation.

    ### 🧠 Aptitude Skills
    These reflect your natural strengths and are measured across:
    - Numerical, Spatial, Perceptual, Abstract, and Verbal reasoning.

    Use the navigation menu on the left to explore the data and test your fit!
    """)

# --- EDA Page ---
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df = pd.read_csv("Data_final.csv")

    st.subheader("📈 Career Count Distribution")
    career_counts = df['Career'].value_counts().reset_index()
    career_counts.columns = ['Career', 'Count']
    fig_pie = px.pie(career_counts, values='Count', names='Career', title='Career Share', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📌 Average Trait Scores by Career")
    selected_trait = st.selectbox("Select a trait to compare by career:", df.select_dtypes(include=np.number).columns)
    avg_scores = df.groupby('Career')[selected_trait].mean().reset_index().sort_values(by=selected_trait)
    fig_bar = px.bar(avg_scores, x=selected_trait, y='Career', orientation='h', color='Career', title=f"Average {selected_trait} by Career")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📉 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Model & Prediction Page ---
elif page == "Model & Prediction":
    st.title("🔍 Predict Your Career")
    st.image("https://img.freepik.com/free-vector/choose-career-concept-illustration_114360-5055.jpg", width=500)
    st.info("Use the sliders to enter your scores. Click 'Predict Career' to get a result.")

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

    - ✅ Clean and interactive dashboard with visual insights
    - 🧠 Random Forest used for robust prediction
    - 📊 Visualizations driven by Plotly and Seaborn

    We hope this project inspires you to explore data-driven career counseling tools.

    **Thank you for using the app!** 💼
    """)
    st.balloons()
