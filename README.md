# 🚀 Insurance Cost Prediction with Neural Networks  

This repository contains a **deep learning model** built using **TensorFlow** and **Keras** to predict insurance costs based on customer data. The model is trained on a dataset with features like age, BMI, smoking status, and region.

## 📌 Features  
- Preprocesses data using **MinMaxScaler** (for numerical features) and **OneHotEncoder** (for categorical features).  
- Implements a **fully connected neural network** using TensorFlow/Keras.  
- Optimized using **Mean Squared Error (MSE)** loss and **Stochastic Gradient Descent (SGD) / Adam** optimizer.  
- Evaluates performance using **Mean Absolute Error (MAE) and R² Score**.  

---

## 📂 Dataset  
The model is trained on an insurance dataset with the following features:  
- `age` (numerical)  
- `bmi` (numerical)  
- `children` (numerical)  
- `sex` (categorical)  
- `smoker` (categorical)  
- `region` (categorical)  
- `charges` (target variable)  

---

## 📜 Installation & Usage  

### 🔧 **1. Clone the Repository**
```bash
git clone https://github.com/Kheera640/insurance-cost-prediction.git
cd insurance-cost-prediction
