# Car Price Prediction using Machine Learning 🚗💡

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Project Overview
This repository contains an end-to-end Machine Learning project designed to estimate the market value of used cars. By analyzing key vehicle attributes—such as brand, model, year of production, mileage, and fuel type—the model provides accurate price predictions. 

The project demonstrates a complete data science pipeline, encompassing data aggregation, thorough preprocessing, feature engineering, model training, evaluation, and deployment through a graphical user interface (GUI).

## 🚀 Key Features
* **Comprehensive Data Processing:** Aggregates and cleans raw data from multiple major car manufacturers (Audi, BMW, Ford, Hyundai, Mercedes, Opel, Skoda, Toyota, VW) into a unified, reliable dataset.
* **Pre-trained ML Model:** Includes a serialized, high-performing predictive model (`final_model.pkl`) ready for immediate inference without the need for retraining.
* **Graphical User Interface (GUI):** Features a user-friendly application (`arayuz.py`) that allows users to input car specifications and receive instant price estimates, making the model accessible to non-technical end-users.

## 🛠️ Technology Stack
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Model Serialization:** Pickle
* **GUI Development:** Tkinter / PyQt (Depending on the implementation in `arayuz.py`)

## 📁 Repository Structure
```text
├── aiproject (1).py                 # Main script for data preprocessing, model training, and evaluation
├── arayuz.py                        # Source code for the Graphical User Interface
├── final_model.pkl                  # Exported pre-trained machine learning model
├── all_cars_merged_no_brand.csv     # Intermediate processed dataset
├── updated_all_cars_with_brand.csv  # Final comprehensive dataset used for training
├── audi.csv, bmw.csv, ...           # Raw dataset files categorized by car brand
└── README.md                        # Project documentation
