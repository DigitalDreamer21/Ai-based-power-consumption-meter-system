import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .forms import PredictionForm

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Globals
home_models, scalers, df = None, None, None

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠ File is empty: {file_path}")
            return None

        df.dropna(subset=['Home Number'], inplace=True)
        df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
        df['Humidity'].fillna(df['Humidity'].mean(), inplace=True)

        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M", errors='coerce')
        df.drop(columns=['Id', 'Status', 'Date', 'Time'], errors='ignore', inplace=True)
        df = df[df['Datetime'].notna()]
        df.fillna(0, inplace=True)

        return df

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def train_models():
    global home_models, scalers, df

    file_paths = [
        "ESP32_Readings_2024-12.csv",
        "ESP32_Readings_2025-01.csv",
        "ESP32_Readings_2025-02.csv",
        "ESP32_Readings_2025-03.csv"
    ]
    dataframes = [load_and_clean_data(fp) for fp in file_paths]
    valid_dfs = [d for d in dataframes if d is not None and not d.empty]

    df = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else None
    if df is None or df.empty:
        print("⚠ No valid data found. Skipping model training.")
        home_models, scalers = {}, {}
        return

    home_models, scalers = {}, {}

    for home in df['Home Number'].unique():
        home_data = df[df['Home Number'] == home].copy()
        if not pd.api.types.is_datetime64_any_dtype(home_data['Datetime']):
            print(f"⚠ Invalid datetime for Home {home}. Skipping...")
            continue

        home_data['Year'] = home_data['Datetime'].dt.year
        home_data['Month'] = home_data['Datetime'].dt.month
        home_data['Day'] = home_data['Datetime'].dt.day
        home_data['Hour'] = home_data['Datetime'].dt.hour
        home_data.drop(columns=['Datetime', 'Home Number'], inplace=True)

        if 'Reading' not in home_data:
            print(f"⚠ Missing 'Reading' for Home {home}. Skipping...")
            continue

        X = home_data.drop(columns=['Reading'])
        y = home_data['Reading']
        X.fillna(0, inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)
        model.fit(X_train, y_train)

        home_models[home] = model
        scalers[home] = scaler

    print(f"✅ Models trained for {len(home_models)} homes.")

def ensure_models_loaded():
    if home_models is None or not home_models:
        train_models()

def home(request):
    form = PredictionForm()
    return render(request, 'home.html', {'form': form})

def predict(request):
    ensure_models_loaded()

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            home_number = form.cleaned_data['home_number']
            date_str = form.cleaned_data['date']
            time_obj = form.cleaned_data['time']

            date = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(date):
                return JsonResponse({'error': 'Invalid date format.'}, status=400)

            hour = time_obj.hour
            model = home_models.get(home_number)
            scaler = scalers.get(home_number)

            if model is None or scaler is None:
                return JsonResponse({'error': f'Home {home_number} not found.'}, status=404)

            # Default inputs
            temp = df['Temperature'].mean() if df is not None else 25
            hum = df['Humidity'].mean() if df is not None else 50

            input_features = pd.DataFrame({
                'Temperature': [temp],
                'Humidity': [hum],
                'Year': [date.year],
                'Month': [date.month],
                'Day': [date.day],
                'Hour': [hour]
            })

            input_scaled = scaler.transform(input_features)
            prediction = model.predict(input_scaled)[0]

            # Plot setup
            home_df = df[df['Home Number'] == home_number]
            actual_data = home_df[home_df['Datetime'].dt.date == date.date()]

            plt.figure(figsize=(8, 4))
            if not actual_data.empty:
                plt.plot(actual_data['Datetime'], actual_data['Reading'], marker='o', label='Actual Readings')
            plt.axhline(prediction, color='red', linestyle='--', label='Predicted Reading')

            plt.title(f"Meter Reading for Home {home_number} on {date.date()}")
            plt.xlabel("Time")
            plt.ylabel("Reading")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save to static/plots/
            plot_id = str(uuid.uuid4())
            plot_filename = f"{plot_id}.png"
            plot_dir = os.path.join(settings.BASE_DIR, 'static', 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()

            return render(request, 'predict.html', {
                'prediction': round(prediction, 2),
                'home_number': home_number,
                'date': date_str,
                'time': time_obj.strftime('%H:%M'),
                'plot_url': f'/static/plots/{plot_filename}'
            })

    return render(request, 'home.html', {'form': PredictionForm()})
