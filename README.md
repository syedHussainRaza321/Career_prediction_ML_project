# 🧠 Career Prediction App (Based on Personality & Aptitude)

Welcome to the Career Prediction App — a machine learning-powered web application that helps predict a suitable career path for users based on their **OCEAN personality traits** and **aptitude scores**.

This project is built with **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**, and is designed to help students or individuals receive career guidance based on psychological and cognitive profiling.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Dataset Description](#-dataset-description)
- [OCEAN Personality Traits](#-ocean-personality-traits)
- [EDA & Insights](#-eda--insights)
- [Modeling Approach](#-modeling-approach)
- [How to Run Locally](#-how-to-run-locally)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 🚀 Demo

🖥️ **Live Streamlit App**: [Click to Launch](https://huggingface.co/spaces/your-app-url)  
📂 **Dataset**: `career_data.csv`

---

## 📊 Dataset Description

This dataset contains records of users along with:

- **Aptitude Score** (0-100)
- **OCEAN Scores**:
  - **Openness**
  - **Conscientiousness**
  - **Extraversion**
  - **Agreeableness**
  - **Neuroticism**
- **Predicted Career Field** (Label)

---

## 🧠 OCEAN Personality Traits

| Trait             | Description |
|------------------|-------------|
| **Openness**      | Creativity, imagination, open to new experiences |
| **Conscientiousness** | Discipline, organization, goal-oriented |
| **Extraversion**   | Sociability, energy, talkativeness |
| **Agreeableness**  | Trust, kindness, affection |
| **Neuroticism**    | Emotional instability, anxiety, moodiness |

Understanding these traits is key in mapping individuals to the right careers.

---

## 📈 EDA & Insights

The app provides rich **Exploratory Data Analysis (EDA)** using:

- Distribution plots
- Correlation heatmaps
- Bar charts showing dominant careers by personality types
- Interactive visuals in Streamlit

---

## 🧠 Modeling Approach

- **Preprocessing**:
  - StandardScaler for normalization
  - Label encoding for career classes
- **Model**: Random Forest Classifier (best accuracy found)
- **Metrics**:
  - Accuracy
  - Classification Report
  - Confusion Matrix
- **Model Files**:
  - `rf_model.pkl`
  - `scaler.pkl`
  - `label_mapping.pkl`

---

## 🛠️ How to Run Locally

1. **Clone this repository**:
   ```bash
   git clone https://github.com/syedHussainRaza321/career-prediction-app.git
   cd career-prediction-app
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
📁 Project Structure
kotlin
Copy
Edit
career-prediction-app/
│
├── data/
│   └── career_data.csv
│
├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── label_mapping.pkl
│
├── pages/
│   ├── 1_📊_EDA.py
│   ├── 2_🧠_Model_&_Prediction.py
│   └── 3_✅_Conclusion.py
│
├── app.py
├── requirements.txt
└── README.md
🖼️ Screenshots
Home Page	EDA Page	Model Output

🧾 Conclusion
This project demonstrates how psychometric profiling + ML can guide career decisions. With further tuning and additional features (like interests, academic scores), the app can serve as a real-world counselor.

👨‍💻 Author
Syed Hussain Raza
🔗 GitHub: @syedHussainRaza321
