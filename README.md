# 🧠 Student Mental Health Assessment System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Machine Learning](https://img.shields.io/badge/ML-Hybrid%20Model-green)
![Status](https://img.shields.io/badge/Status-Defense%20Ready-success)

## 📌 Project Overview
This project is a **Machine Learning-based Screening System** designed to assess the mental health status of university students. Unlike traditional rule-based forms, this system utilizes a **Hybrid Ensemble Model (Random Forest + XGBoost)** to analyze behavioral patterns and predict risk levels for **Anxiety, Depression, and Stress**.

The system is designed with a **"Safety-First"** and **"User-Centric"** approach, featuring bilingual support (English/Bangla), strict data validation, and immediate emergency alerts for high-risk individuals.

---

## 🚀 Key Features

### 1. 🛡️ Strict Profile Gatekeeping
- The assessment is **locked** by default.
- Users must complete a valid **Student Profile** (Age, Gender, CGPA, Dept, etc.) and confirm their details to unlock the questionnaire.
- This ensures **Data Integrity** and prevents noisy/lazy inputs.

### 2. 🤖 Hybrid Machine Learning Engine
- **Algorithm:** Ensemble of Random Forest & XGBoost.
- **Input:** 33 Features (7 Demographic + 26 Behavioral).
- **Output:** Probabilistic risk assessment (not just a raw score).

### 3. 🌍 Bilingual Support (Localization)
- Full interface support for **English** and **Bangla**.
- Ensures accessibility for students from diverse backgrounds while maintaining English-coded values for the ML backend.

### 4. 🚨 Safety Net & Emergency Protocols
- **Q26 Safety Override:** If a user indicates self-harm tendencies (Question 26), the system triggers an immediate **Red Emergency Alert** and provides helpline numbers, regardless of the ML prediction.
- **Clinical Note:** If the ML predicts "Healthy" but specific risk indicators are high, a clinical warning is displayed inside the result card.

### 5. 💡 Friendly vs. Clinical Suggestions
- **Mild/Moderate Cases:** The system provides casual, family-friendly, and actionable daily tips.
- **Severe Cases:** The tone shifts to professional, recommending psychologists or counselors.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Python)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Plotly
- **Serialization:** Joblib

---

## 📂 Project Structure

```text
├── app.py                        # Main Application Code
├── requirements.txt              # Dependency List
├── mental_health_hybrid_model.pkl # Trained Hybrid ML Model
├── label_encoders.pkl            # Encoders for Categorical Data
├── feature_columns.pkl           # Feature Alignment Object
└── README.md                     # Project Documentation

⚙️ Installation & Setup
Clone the Repository

Bash

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
Install Dependencies

Bash

pip install -r requirements.txt
Run the Application

Bash

streamlit run app.py
📝 Usage Workflow
Profile Entry: Fill in your student details on the left sidebar.

Unlock: Click "Save & Start Assessment".

Assessment: Answer the 26 behavioral questions using the radio buttons.

Analysis: Click "Analyze My Mental Health".

Report: View your risk levels, get tailored suggestions, and download the Wellness Report.

⚠️ Disclaimer
This application is a research prototype and a screening tool. It uses probabilistic machine learning patterns to estimate risk levels. It is NOT a substitute for a professional clinical diagnosis. If you are feeling overwhelmed, please consult a certified mental health professional.

👨‍💻 Developed By
Team Dual Core Department of CSE, Daffodil International University
