⚡ AI-Based Power Consumption Meter System
📌 Project Overview

The AI-Based Power Consumption Meter System is an intelligent energy monitoring solution that combines IoT (ESP32) and Machine Learning to track, analyze, and predict power consumption in real time.
It helps users understand energy usage patterns and enables data-driven decisions for power optimization.

🎯 Problem Statement

Traditional power meters only display consumption values and lack:

Predictive insights

Usage pattern analysis

Smart decision support

This project addresses these limitations by using AI models trained on real sensor data to predict future power consumption and visualize trends.

🚀 Features

📡 Real-time power data collection using ESP32

📊 Data storage in CSV & SQLite database

🤖 Machine Learning–based power prediction

🌐 Web dashboard for visualization

📈 Graphs for daily, monthly, and yearly analysis

🔄 Scalable for multiple locations (Home A, Home B, Lab, etc.)

🛠️ Tech Stack
Hardware

ESP32 Microcontroller

Voltage & Current Sensors

Backend

Python

Django

SQLite

Machine Learning

Scikit-Learn

NumPy

Pandas

Frontend

HTML

CSS

Django Templates

Tools

Git & GitHub

Matplotlib

🏗️ Project Structure
├── powermeter/                 # Django application
├── templates/                  # HTML templates
├── static/plots/               # Generated power graphs
├── ESP32_Readings_*.csv        # Sensor data files
├── db.sqlite3                  # Database
├── manage.py                   # Django entry point
├── model_*.pkl                 # Trained ML models
├── scaler_*.pkl                # Feature scalers
└── README.md                   # Project documentation

🔁 System Architecture

ESP32 collects voltage & current data

Data stored in CSV and SQLite database

Data preprocessing and scaling

ML model predicts power consumption

Web app displays predictions & graphs

🤖 Machine Learning Workflow

Data Cleaning & Normalization

Feature Scaling using StandardScaler

Model Training (Regression Models)

Model Serialization using .pkl

Real-time Prediction via Backend

▶️ How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/DigitalDreamer21/Ai-based-power-consumption-meter-system.git
cd Ai-based-power-consumption-meter-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Django Server
python manage.py runserver

4️⃣ Open in Browser
http://127.0.0.1:8000/

📊 Results

Accurate prediction of power consumption trends

Clear visualization of energy usage

Improved understanding of peak and off-peak consumption
