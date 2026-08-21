# 🧒 Stunting Risk Classification using K-Nearest Neighbors

A machine learning project that classifies children's stunting risk based on anthropometric and demographic data, deployed as an interactive single-page web application.

---

## 📖 Table of Contents
* [About the Project](#-about-the-project)
* [Objectives](#-objectives)
* [Dataset & Preprocessing](#-dataset--preprocessing)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Web Application Interface](#-web-application-interface)
* [Technologies](#-technologies)
* [Project Structure](#-project-structure)
* [How to Run](#-how-to-run)
* [Project Highlights](#-project-highlights)
* [Author](#-author)
* [Disclaimer](#-disclaimer)

---

## 📌 About the Project
**Stunting Risk Classification** is a machine learning project developed to predict the risk of stunting in children based on health and anthropometric features.

The project applies a classification approach comparing **K-Nearest Neighbors (KNN)** and **Random Forest**. The best-performing model (KNN) is deployed as a single-page interactive web application where users can input child metrics and instantly receive real-time nutritional risk predictions.

---

## 🎯 Objectives
* Analyze factors associated with stunting risk in children.
* Perform data preprocessing, cleaning, and exploratory data analysis (EDA).
* Apply and compare Machine Learning classification algorithms (KNN vs. Random Forest).
* Evaluate classification performance using standard evaluation metrics.
* Deploy the final model into a lightweight, user-friendly single-page web application.

---

## 📊 Dataset & Preprocessing

* **Target Variable:** Stunting risk / nutritional status classification (*e.g., Normal, Stunted*).
* **Key Features:** Age (months), Height (cm), and Weight (kg).

### Preprocessing Steps:
* Data cleaning and handling missing/inconsistent values.
* Feature scaling and transformation for distance-based classification.
* Stratified train-test split for model validation.

---

## 🤖 Machine Learning Models

* **1. K-Nearest Neighbors (KNN) — Selected Model**  
  Classifies child status based on proximity metrics to historical observations. Selected as the production model due to its high precision and generalization.
* **2. Random Forest**  
  An ensemble baseline model used to benchmark and compare classification accuracy.

---

## 📈 Model Evaluation

The KNN model was evaluated using standard classification metrics:

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **94%** |
| **Precision** | **90%** |
| **Recall** | **91%** |
| **F1-Score** | **91%** |

---

## 🖥️ Web Application Interface

The trained KNN model is integrated into a single-page web interface. Users enter the child's parameters (Name, Age, Height, Weight), submit the form, and immediately receive the classification result on the same page.

<img width="1600" height="845" alt="Dashboard stunting" src="https://github.com/user-attachments/assets/1d2c5b1b-c84f-493b-bfdf-741eb19a9ace" />


---

## 🛠️ Technologies

| Category | Tools / Libraries |
| :--- | :--- |
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 📁 Project Structure
```text

stunting-knn/
│
├── images/
│   └── dashboard.png
│
├── notebook/
│   └── stunting_knn.ipynb
│
├── app.py
├── requirements.txt
└── README.md<img width="1600" height="845" alt="Dashboard stunting" src="https://github.com/user-attachments/assets/be762aca-0fb0-4f40-b8e8-ee4b551a5f86" />
```
.

---



## 👤 Author

**M. Maulud Syafrizal**

Fresh Graduate — S1 Informatika
Universitas Amikom Yogyakarta
Interests: Data Science, Machine Learning, Data Analysis
