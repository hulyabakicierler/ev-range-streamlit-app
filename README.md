#  EV Range & Segment Prediction App (Streamlit)

Predict electric vehicle range and segment class using machine learning and Streamlit.


This application allows users to **predict the range** and **segment class** of electric vehicles (EVs) based on various technical specifications. It was developed using Python, machine learning (XGBoost & Random Forest), and deployed via Streamlit.

---

##  Features

-  Predict estimated **range (in km)** using regression (XGBoost)
-  Predict **vehicle segment class** (e.g., Compact, SUV, Luxury) using classification (Random Forest)
-  User-friendly interface built with **Streamlit**
- ⚙ Includes preprocessing steps such as **scaling** and **feature engineering**

---

##  Files Included

- `app.py`: Main Streamlit application
- `xgb_model.pkl`: Trained XGBoost regression model for range prediction
- `rf_model.pkl`: Trained Random Forest classifier for vehicle segmentation
- `scaler.pkl`: Scikit-learn scaler object used for preprocessing
- `label_encoder.pkl`: Label encoder for vehicle segment labels

---

## 🛠 How It Works

1. The user inputs key vehicle attributes:
   - Battery capacity, efficiency, top speed, torque, acceleration, etc.
2. The app calculates volume and estimated range.
3. The input is scaled using the same scaler used during training.
4. Two models are used:
   - `XGBoost` predicts the estimated driving range.
   - `Random Forest` predicts the vehicle’s segment class.
5. The results are displayed dynamically.

---

##  Requirements

To run the app locally:

```bash
pip install streamlit xgboost scikit-learn pandas joblib
