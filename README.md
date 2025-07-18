# 💼 Customer Churn Prediction App

A Streamlit web application that predicts whether a bank customer is likely to **churn** or stay, based on their profile.

This project uses a trained **Artificial Neural Network (ANN)** built with **Keras**, and serves predictions through a modern, responsive UI powered by **Streamlit**.

---

## ✨ Features

- 🔢 Input customer details (credit score, age, geography, etc.)
- 📈 Uses a trained ANN model for prediction
- 📉 Probability-based output with churn classification
- ⚙️ Model and scaler preloaded for instant inference
- 💡 Clean UI layout using `st.columns` and metrics
- ☁️ Ready for deployment on **Streamlit Community Cloud**

---

## 🧠 Tech Stack

| Component       | Tool / Library            |
|-----------------|---------------------------|
| Interface       | `Streamlit`               |
| Model Training  | `TensorFlow` / `Keras`    |
| Preprocessing   | `scikit-learn`, `pandas`  |
| Deployment      | `Streamlit Cloud`         |

---

## 🗂 Folder Structure

````
Churn Prediction Model/
├── app.py                   # Streamlit UI code
├── model/
│   ├── churn\_ann\_model.h5   # Trained ANN model
│   └── scaler.pkl           # Saved StandardScaler
├── requirements.txt         # Required Python packages
````

---

## ⚙️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
````

2. **Set up virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

3. **Install requirements**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 📦 `requirements.txt`

```txt
streamlit
tensorflow
scikit-learn
pandas
numpy
```

---

## 📌 Example Use Case

| Input                       | Value        |
| --------------------------- | ------------ |
| Age                         | 42           |
| Geography                   | Germany      |
| Balance                     | 50,000.00    |
| Products                    | 2            |
| Is Active Member?           | Yes          |
| Estimated Churn Probability | 73.2%        |
| Result                      | ❌ Will Churn |

---

## 📜 License

MIT © 2025 \ Lakshay Jain

---

## 🙋‍♂️ Author

Built by **Lakshay Jain** as a foundational deep learning project