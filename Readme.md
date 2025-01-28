# ⚡ Electricity Load Forecasting Project 🔮

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

![Cover Image](./Image/power_line.jpg)

## WARNING: THIS PROJECT IS CONFIGURED FOR MACBOOK. PLEASE CHECK LIBRARY AND SYSTEM COMPATIBILITY BEFORE RUNNING.

This project focuses on forecasting electricity demand using a combination of deep learning and machine learning models. The models implemented include PyTorch LSTM, Keras LSTM, Linear Regression, Gradient Boosting, LightGBM, ARIMA, and SARIMA. The project includes extensive evaluations, visualizations, and insights to compare model performances and optimize forecasting accuracy.

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [📊 Dataset](#dataset)
3. [💻 Installation](#installation)
4. [🚀 Usage](#usage)
5. [🧠 Models](#models)
6. [📈 Evaluation and Findings](#evaluation-and-findings)
7. [📉 Results and Visualizations](#results-and-visualizations)
8. [🔮 Future Improvements](#future-improvements)
9. [📂 Project Structure](#project-structure)
10. [🙏 Acknowledgments](#acknowledgments)
11. [📝 License](#license)

---

## Introduction

Accurate forecasting of electricity demand is crucial for energy management, cost efficiency, and grid stability. This project leverages both deep learning and traditional machine learning models to analyze historical electricity load data and generate accurate demand predictions.

### 🎯 Objectives

- **Compare Model Performance:** Evaluate different forecasting models for electricity demand prediction.
- **Generate Visual Insights:** Use advanced data visualizations to understand prediction patterns.
- **Improve Forecasting Accuracy:** Implement and test techniques to enhance prediction reliability.

---

## 📊 Dataset

The dataset used in this project is sourced from Kaggle:

- **[Electricity Load Forecasting Dataset](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting)** by [Saurabh Shahane](https://www.kaggle.com/saurabhshahane).

### 📝 Dataset Details

- **Time Interval:** Hourly electricity load data.
- **Features:** Includes time-based attributes, weather conditions, and categorical indicators (e.g., holidays, weekdays).
- **Target Variable:** `DEMAND` - representing electricity load.

---

## 💻 Installation

To run this project on your local machine, follow the steps below.

### Prerequisites

- **Operating System:** macOS (tested) or Linux (adjustments may be needed for Windows).
- **Python:** Version 3.12+
- **Git:** Required to clone the repository.

### Clone the Repository

```bash
git clone https://github.com/Btry123/electricity-forecasting.git
cd electricity-forecasting
```

### 🛠️ Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv electricity_load_env

# Activate the virtual environment
# macOS/Linux:
source electricity_load_env/bin/activate

# Windows:
electricity_load_env\Scripts\activate
```

### 📦 Install Required Libraries

```bash
pip install -r requirements.txt
```

### 📂 Data Setup

1. **Download the Dataset:**
   - Visit Kaggle and download the dataset.
2. **Organize Data Files:**
   - Extract files and place them in the `archive/` directory.
3. **Verify File Paths:**
   - Ensure the dataset file paths in the scripts match the actual file locations.

---

## 🚀 Usage

### Running the Project

1. **Activate Virtual Environment:**
   ```bash
   source electricity_load_env/bin/activate  # macOS/Linux
   electricity_load_env\Scripts\activate  # Windows
   ```
2. **Execute the Main Script:**
   ```bash
   python main.py
   ```

This will run the full pipeline including data preprocessing, model training, evaluation, and visualization.

### Using Jupyter Notebooks

To interactively explore the data and test models, use:

```bash
jupyter notebook
```

and open `notebooks/main.ipynb` or `notebooks/playground.ipynb`.

---

## 🧠 Models

The following forecasting models are implemented:

1. **PyTorch LSTM** – Deep learning model designed to capture long-term dependencies in sequential data.
2. **Keras LSTM** – Similar to PyTorch LSTM but implemented using Keras/TensorFlow.
3. **Linear Regression** – A simple statistical model establishing relationships between features and demand.
4. **Gradient Boosting** – An ensemble learning technique that builds multiple models iteratively to correct errors.
5. **LightGBM** – A tree-based boosting method optimized for performance and efficiency.
6. **ARIMA** – Traditional statistical time-series model for analyzing trends and dependencies.
7. **SARIMA** – An extension of ARIMA that accounts for seasonal patterns.

Each model is trained, validated, and tested to compare their effectiveness in predicting electricity demand.

---

## 📈 Evaluation and Findings

### Metrics Used

- **RMSE (Root Mean Square Error):** Measures the magnitude of prediction errors.
- **MAE (Mean Absolute Error):** Represents the average absolute difference between predicted and actual values.
- **MAPE (Mean Absolute Percentage Error):** Measures accuracy as a percentage.

### Observations

- **PyTorch LSTM** achieved the best performance with RMSE = **39.85**, MAPE = **2.86%**.
- **LightGBM and Gradient Boosting** provided strong alternatives with RMSE = **51.09** and **53.93**, respectively.
- **Keras LSTM** performed moderately well with RMSE = **95.20**.
- **Linear Regression** and **SARIMA** struggled with capturing complex relationships, showing RMSEs of **100.26** and **134.56**.
- **ARIMA** had the highest RMSE = **283.74**, indicating it was unsuitable for this dataset.

---

## 📉 Results and Visualizations

### Visualizations

- **Hourly Predictions:** Model performance for each hour.
- **Daily Aggregated Trends:** Smoothed visualizations of daily demand.
- **Model Comparison Charts:** Plots comparing actual vs. predicted values for all models.

### Viewing Results

- **Graphs:** Generated plots are saved in `visualizations/`.
- **Metrics:** Printed in console logs and Jupyter notebooks.

---

## 🔮 Future Improvements

To further enhance accuracy and efficiency, the following improvements can be explored:

1. **Feature Engineering:** Adding external factors like weather conditions to enrich input features.
2. **Automated Hyperparameter Optimization:** Using Grid Search or Bayesian Optimization.
3. **Model Ensembling:** Combining multiple models for improved predictions.
4. **Deploying Real-Time Forecasting:** Implementing an API for real-time electricity load predictions.

---

## 📂 Project Structure

```
electricity-forecasting/
├── archive/
│   ├── train_dataframes.xlsx
│   ├── test_dataframes.xlsx
├── visualizations/
│   ├── hourly_predictions.png
│   ├── daily_predictions.png
├── notebooks/
│   ├── main.ipynb
│   ├── playground.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🙏 Acknowledgments

- **Dataset:** Thanks to Saurabh Shahane for the dataset on Kaggle.
- **Open-Source Community:** Credits to contributors of PyTorch, TensorFlow, LightGBM, and scikit-learn.

---

## 📝 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it as per the license terms.

