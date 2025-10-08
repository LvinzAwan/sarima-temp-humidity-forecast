from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
import numpy as np
from typing import Optional, List
import warnings
import requests
from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Prediksi Suhu & Kelembapan API",
    description="API untuk prediksi suhu dan kelembapan menggunakan model SARIMA",
    version="1.0.0"
)

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],      
)

class SensorDataPoint(BaseModel):
    timestamp: str = Field(..., description="Timestamp format ISO (YYYY-MM-DDTHH:MM:SS)")
    suhu: float = Field(..., description="Nilai suhu dalam Celsius")
    kelembapan: float = Field(..., description="Nilai kelembapan dalam persen (%)")

class PredictionRequest(BaseModel):
    lokasi: str = Field(..., description="Lokasi sensor:  kulkas")
    data_sensor: List[SensorDataPoint] = Field(..., description="Data sensor historis (minimal 3 data point)")
    steps: Optional[int] = Field(default=3, description="Jumlah jam prediksi (1-12)")

class PredictionItem(BaseModel):
    jam_ke: int
    suhu: float
    kelembapan: float

class PredictionResponse(BaseModel):
    lokasi: str
    data_points_used: int
    strategy_used: str
    anomaly_detected: bool
    note: Optional[str] = None
    prediksi: List[PredictionItem]

models_cache = {}
model_stats = {}


def download_model_from_hf(lokasi: str):
    model_repo = "pauluswindi/prediction-sarima-models"
    model_url = f"https://huggingface.co/{model_repo}/resolve/main/{lokasi}_models.pkl"
    model_path = f"/tmp/{lokasi}_models.pkl"
    
    try:
        print(f"Downloading {lokasi} from {model_url}")
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        print(f"Model {lokasi} saved to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model {lokasi}: {str(e)}")
        return None

def load_models():
    global models_cache, model_stats
    lokasi_list = ['kulkas']
    
    for lokasi in lokasi_list:
        model_file = f"tmp/{lokasi}_models.pkl"
        
        if not os.path.exists(model_file):
            model_file = download_model_from_hf(lokasi)
        
        if model_file and os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                print(f"Loaded data structure for {lokasi}: {type(loaded_data)}")
                print(f"Keys in loaded data: {list(loaded_data.keys()) if isinstance(loaded_data, dict) else 'Not a dict'}")
                
                # Check if the structure has all required components
                if isinstance(loaded_data, dict) and 'models' in loaded_data and 'scalers' in loaded_data:
                    actual_models = loaded_data['models']
                    scalers = loaded_data['scalers']
                    
                    print(f"Models structure for {lokasi}: {type(actual_models)}")
                    print(f"Keys in models: {list(actual_models.keys()) if isinstance(actual_models, dict) else 'Not a dict'}")
                    print(f"Scalers available: {list(scalers.keys()) if isinstance(scalers, dict) else 'Not a dict'}")
                    
                    # Check if suhu and kelembapan models exist
                    if (isinstance(actual_models, dict) and 'suhu' in actual_models and 'kelembapan' in actual_models and
                        isinstance(scalers, dict) and 'suhu' in scalers and 'kelembapan' in scalers):
                        
                        models_cache[lokasi] = {
                            'models': actual_models,
                            'scalers': scalers,
                            'metrics': loaded_data.get('metrics', {}),
                            'data_range': loaded_data.get('data_range', None)
                        }
                        print(f"✓ Model {lokasi} loaded successfully with scalers")
                        model_stats[lokasi] = get_default_model_stats(lokasi)
                    else:
                        print(f"✗ Model {lokasi} missing required components")
                        print(f"  Models keys: {list(actual_models.keys()) if isinstance(actual_models, dict) else 'None'}")
                        print(f"  Scalers keys: {list(scalers.keys()) if isinstance(scalers, dict) else 'None'}")
                
                else:
                    print(f"✗ Model {lokasi} format not recognized. Missing 'models' or 'scalers' key")
                    if isinstance(loaded_data, dict):
                        print(f"Available keys: {list(loaded_data.keys())}")
                    
            except Exception as e:
                print(f"✗ Error loading model {lokasi}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ Model {lokasi} file not available")

def get_default_model_stats(lokasi: str) -> dict:
    stats_mapping = {
        'kulkas': {
            'suhu': {'mean': 4.0, 'std': 1.0, 'min': 1.0, 'max': 8.0},
            'kelembapan': {'mean': 85.0, 'std': 5.0, 'min': 75.0, 'max': 95.0}
        }
    }
    return stats_mapping.get(lokasi, {
        'suhu': {'mean': 25.0, 'std': 3.0, 'min': 15.0, 'max': 35.0},
        'kelembapan': {'mean': 60.0, 'std': 10.0, 'min': 40.0, 'max': 80.0}
    })

def scale_data(data, column, scaler):
    """Scale data using the provided scaler"""
    return scaler.transform(data[[column]]).flatten()

def inverse_scale_data(values, column, scaler):
    """Inverse scale data using the provided scaler"""
    if isinstance(values, pd.Series):
        values = values.values
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()

def detect_anomaly_and_choose_strategy(data: pd.DataFrame, lokasi: str) -> tuple[str, bool, str]:
    stats = model_stats.get(lokasi, {})
    suhu_stats = stats.get('suhu', {})
    kelembapan_stats = stats.get('kelembapan', {})
    
    current_suhu_mean = data['suhu'].mean()
    current_kelembapan_mean = data['kelembapan'].mean()
    current_suhu_max = data['suhu'].max()
    current_suhu_min = data['suhu'].min()
    current_kelembapan_max = data['kelembapan'].max()
    current_kelembapan_min = data['kelembapan'].min()
    
    anomaly_reasons = []
    
    # Make anomaly detection less sensitive by using 3.0 instead of 2.5 standard deviations
    suhu_lower = suhu_stats.get('mean', 25) - (3.0 * suhu_stats.get('std', 3))
    suhu_upper = suhu_stats.get('mean', 25) + (3.0 * suhu_stats.get('std', 3))
    
    if current_suhu_min < suhu_lower or current_suhu_max > suhu_upper:
        anomaly_reasons.append(f"Suhu di luar rentang normal ({suhu_lower:.1f}°C - {suhu_upper:.1f}°C)")
    
    kelembapan_lower = kelembapan_stats.get('mean', 60) - (3.0 * kelembapan_stats.get('std', 10))
    kelembapan_upper = kelembapan_stats.get('mean', 60) + (3.0 * kelembapan_stats.get('std', 10))
    kelembapan_lower = max(0, kelembapan_lower)
    kelembapan_upper = min(100, kelembapan_upper)
    
    if current_kelembapan_min < kelembapan_lower or current_kelembapan_max > kelembapan_upper:
        anomaly_reasons.append(f"Kelembapan di luar rentang normal ({kelembapan_lower:.1f}% - {kelembapan_upper:.1f}%)")
    
    if len(data) >= 2:
        suhu_changes = abs(data['suhu'].diff().dropna())
        kelembapan_changes = abs(data['kelembapan'].diff().dropna())
        
        max_suhu_change = suhu_changes.max()
        max_kelembapan_change = kelembapan_changes.max()
        
        # Increased thresholds to be less sensitive
        if max_suhu_change > 7.0:  # was 5.0
            anomaly_reasons.append(f"Perubahan suhu drastis: {max_suhu_change:.1f}°C/jam")
        
        if max_kelembapan_change > 15.0:  # was 12.0
            anomaly_reasons.append(f"Perubahan kelembapan drastis: {max_kelembapan_change:.1f}%/jam")
    
    if len(data) >= 3:
        suhu_trend = calculate_trend_strength(data['suhu'].values)
        kelembapan_trend = calculate_trend_strength(data['kelembapan'].values)
        
        # Significantly increased threshold from 0.9 to 0.97 to be much less sensitive
        if abs(suhu_trend) > 0.97:
            anomaly_reasons.append(f"Trend suhu sangat kuat")
        
        if abs(kelembapan_trend) > 0.97:
            anomaly_reasons.append(f"Trend kelembapan sangat kuat")
    
    is_anomaly = len(anomaly_reasons) > 0
    
    # Force hybrid as default unless there's a clear anomaly
    if is_anomaly:
        strategy = "trend_based"
        explanation = f"Prediksi berbasis trend: {'; '.join(anomaly_reasons)}"
    else:
        strategy = "hybrid"
        explanation = "Prediksi hybrid (model + trend) - kondisi normal"
    
    return strategy, is_anomaly, explanation

def calculate_trend_strength(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    
    x = np.arange(len(values))
    correlation = np.corrcoef(x, values)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0

def process_input_data(sensor_data: List[SensorDataPoint]) -> tuple[pd.DataFrame, str]:
    data_list = []
    for point in sensor_data:
        data_list.append({
            'timestamp': point.timestamp,
            'suhu': point.suhu,
            'kelembapan': point.kelembapan
        })
    
    df = pd.DataFrame(data_list)
    
    if len(df) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Minimal 3 data point diperlukan"
        )
    
    note = None
    if len(df) < 6:
        note = f"Data kurang dari 6 jam ({len(df)} point), akurasi mungkin kurang optimal. Disarankan 12+ data point."
    elif len(df) < 12:
        note = f"Data kurang dari 12 jam ({len(df)} point), untuk hasil terbaik gunakan 12+ data point."
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['suhu'] = pd.to_numeric(df['suhu'], errors='coerce')
        df['kelembapan'] = pd.to_numeric(df['kelembapan'], errors='coerce')
        
        if df['suhu'].isna().any() or df['kelembapan'].isna().any():
            df['suhu'] = df['suhu'].interpolate(method='linear')
            df['kelembapan'] = df['kelembapan'].interpolate(method='linear')
            
            if df['suhu'].isna().any() or df['kelembapan'].isna().any():
                raise HTTPException(
                    status_code=400,
                    detail="Data mengandung nilai yang tidak valid"
                )
        
        if df['suhu'].min() < -50 or df['suhu'].max() > 100:
            note = (note or "") + " Peringatan: Nilai suhu di luar rentang normal."
        
        if df['kelembapan'].min() < 0 or df['kelembapan'].max() > 100:
            note = (note or "") + " Peringatan: Nilai kelembapan di luar rentang normal."
        
        return df, note
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error format timestamp atau data numerik: {str(e)}"
        )

def predict_with_input_data(lokasi: str, data: pd.DataFrame, steps: int = 3) -> tuple[List[dict], str, bool, str]:
    strategy, is_anomaly, explanation = detect_anomaly_and_choose_strategy(data, lokasi)
    
    # Check if location exists in models_cache
    if lokasi not in models_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model untuk lokasi '{lokasi}' tidak tersedia"
        )
    
    # Get models and scalers
    model_data = models_cache[lokasi]
    models = model_data['models']
    scalers = model_data['scalers']
    
    if not isinstance(models, dict) or not isinstance(scalers, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Format model atau scaler untuk lokasi '{lokasi}' tidak valid"
        )
    
    if 'suhu' not in models or 'suhu' not in scalers:
        raise HTTPException(
            status_code=500,
            detail=f"Model atau scaler suhu untuk lokasi '{lokasi}' tidak ditemukan"
        )
    
    if 'kelembapan' not in models or 'kelembapan' not in scalers:
        raise HTTPException(
            status_code=500,
            detail=f"Model atau scaler kelembapan untuk lokasi '{lokasi}' tidak ditemukan"
        )
    
    if strategy == "hybrid":
        # Validate models are callable
        try:
            suhu_model = models['suhu']
            kelembapan_model = models['kelembapan']
            
            # Check if models have forecast method
            if not hasattr(suhu_model, 'forecast'):
                strategy = "trend_based"
                explanation += " | Model suhu tidak memiliki method forecast"
            elif not hasattr(kelembapan_model, 'forecast'):
                strategy = "trend_based"
                explanation += " | Model kelembapan tidak memiliki method forecast"
                
        except Exception as e:
            strategy = "trend_based"
            explanation += f" | Error accessing models: {str(e)}"
    
    try:
        data_indexed = data.set_index('timestamp')
        suhu_data = data_indexed['suhu'].dropna()
        kelembapan_data = data_indexed['kelembapan'].dropna()
        
        if strategy == "trend_based":
            predictions = predict_pure_trend(suhu_data, kelembapan_data, steps)
        else:
            try:
                suhu_model = models['suhu']
                kelembapan_model = models['kelembapan']
                suhu_scaler = scalers['suhu']
                kelembapan_scaler = scalers['kelembapan']
                
                trend_weight = 0.6 if is_anomaly else 0.3
                predictions = predict_with_trend_adaptation_scaled(
                    suhu_data, kelembapan_data, 
                    suhu_model, kelembapan_model,
                    suhu_scaler, kelembapan_scaler,
                    steps, trend_weight
                )
            except Exception as model_error:
                print(f"Model prediction failed: {str(model_error)}")
                # Fallback to trend-based prediction
                predictions = predict_pure_trend(suhu_data, kelembapan_data, steps)
                strategy = "trend_based"
                explanation = f"Fallback to trend-based due to model error: {str(model_error)}"
        
        return predictions, strategy, is_anomaly, explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error prediksi: {str(e)}"
        )

def predict_with_trend_adaptation_scaled(suhu_data: pd.Series, kelembapan_data: pd.Series, 
                                       suhu_model, kelembapan_model,
                                       suhu_scaler, kelembapan_scaler,
                                       steps: int, trend_weight: float = 0.3) -> List[dict]:
    """
    Enhanced prediction function that handles data scaling properly
    """
    suhu_trend = calculate_advanced_trend(suhu_data.values)
    kelembapan_trend = calculate_advanced_trend(kelembapan_data.values)
    
    last_suhu = suhu_data.iloc[-1]
    last_kelembapan = kelembapan_data.iloc[-1]
    
    # Scale the input data for model prediction
    try:
        # Create DataFrame for scaling
        temp_df_suhu = pd.DataFrame({'suhu': suhu_data})
        temp_df_kelembapan = pd.DataFrame({'kelembapan': kelembapan_data})
        
        # Scale the data
        suhu_scaled = scale_data(temp_df_suhu, 'suhu', suhu_scaler)
        kelembapan_scaled = scale_data(temp_df_kelembapan, 'kelembapan', kelembapan_scaler)
        
        # Get model forecasts on scaled data
        model_suhu_forecast_scaled = suhu_model.forecast(steps=steps)
        model_kelembapan_forecast_scaled = kelembapan_model.forecast(steps=steps)
        
        # Convert to Series if needed
        if not isinstance(model_suhu_forecast_scaled, pd.Series):
            model_suhu_forecast_scaled = pd.Series(model_suhu_forecast_scaled)
        if not isinstance(model_kelembapan_forecast_scaled, pd.Series):
            model_kelembapan_forecast_scaled = pd.Series(model_kelembapan_forecast_scaled)
        
        # Inverse scale the forecasts to original units
        model_suhu_forecast = inverse_scale_data(model_suhu_forecast_scaled, 'suhu', suhu_scaler)
        model_kelembapan_forecast = inverse_scale_data(model_kelembapan_forecast_scaled, 'kelembapan', kelembapan_scaler)
        
    except Exception as e:
        print(f"Scaling or model forecast failed: {e}")
        raise Exception(f"Error in scaled prediction: {str(e)}")
    
    predictions = []
    
    for i in range(steps):
        # Slower decay of trend weight to maintain model dominance
        current_trend_weight = trend_weight * (0.95 ** i)
        current_model_weight = 1 - current_trend_weight
        
        trend_suhu = last_suhu + (suhu_trend * (i + 1))
        trend_kelembapan = last_kelembapan + (kelembapan_trend * (i + 1))
        
        try:
            final_suhu = (trend_suhu * current_trend_weight) + (float(model_suhu_forecast[i]) * current_model_weight)
            final_kelembapan = (trend_kelembapan * current_trend_weight) + (float(model_kelembapan_forecast[i]) * current_model_weight)
        except Exception as e:
            print(f"Error combining predictions at step {i}: {e}")
            # Fallback to trend only for this step
            final_suhu = trend_suhu
            final_kelembapan = trend_kelembapan
        
        final_suhu = apply_constraints(final_suhu, 'suhu', last_suhu)
        final_kelembapan = apply_constraints(final_kelembapan, 'kelembapan', last_kelembapan)
        
        predictions.append({
            "jam_ke": i + 1,
            "suhu": round(final_suhu, 2),
            "kelembapan": round(final_kelembapan, 2)
        })
    
    return predictions

def predict_pure_trend(suhu_data: pd.Series, kelembapan_data: pd.Series, steps: int) -> List[dict]:
    suhu_trend = calculate_advanced_trend(suhu_data.values)
    kelembapan_trend = calculate_advanced_trend(kelembapan_data.values)
    
    last_suhu = suhu_data.iloc[-1]
    last_kelembapan = kelembapan_data.iloc[-1]
    
    suhu_volatility = calculate_volatility(suhu_data.values)
    kelembapan_volatility = calculate_volatility(kelembapan_data.values)
    
    predictions = []
    
    for i in range(steps):
        damping_factor = 0.95 ** i
        
        pred_suhu = last_suhu + (suhu_trend * (i + 1) * damping_factor)
        pred_kelembapan = last_kelembapan + (kelembapan_trend * (i + 1) * damping_factor)
        
        suhu_noise = np.random.normal(0, suhu_volatility * 0.1)
        kelembapan_noise = np.random.normal(0, kelembapan_volatility * 0.1)
        
        pred_suhu += suhu_noise
        pred_kelembapan += kelembapan_noise
        
        pred_suhu = apply_pure_constraints(pred_suhu, 'suhu', last_suhu, i + 1)
        pred_kelembapan = apply_pure_constraints(pred_kelembapan, 'kelembapan', last_kelembapan, i + 1)
        
        predictions.append({
            "jam_ke": i + 1,
            "suhu": round(pred_suhu, 2),
            "kelembapan": round(pred_kelembapan, 2)
        })
    
    return predictions

def calculate_advanced_trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = np.arange(n)
    weights = np.exp(x / n)
    
    A = np.vstack([x, np.ones(len(x))]).T
    W = np.diag(weights)
    slope, _ = np.linalg.lstsq(W @ A, W @ values, rcond=None)[0]
    
    max_reasonable_trend = np.std(values) * 0.3
    slope = np.clip(slope, -max_reasonable_trend, max_reasonable_trend)
    
    return float(slope)

def calculate_volatility(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.1
    
    diffs = np.diff(values)
    volatility = np.std(diffs)
    
    return max(volatility, 0.05)

def apply_pure_constraints(value: float, param_type: str, last_value: float, step: int) -> float:
    max_hourly_change = {
        'suhu': 1.5,
        'kelembapan': 3.0
    }
    
    max_change_per_step = max_hourly_change.get(param_type, 1.0) * step
    
    if abs(value - last_value) > max_change_per_step:
        if value > last_value:
            value = last_value + max_change_per_step
        else:
            value = last_value - max_change_per_step
    
    if param_type == 'suhu':
        value = np.clip(value, -30, 60)
    elif param_type == 'kelembapan':
        value = np.clip(value, 0, 100)
    
    return value

def apply_constraints(value: float, param_type: str, last_value: float) -> float:
    max_hourly_change = {
        'suhu': 2.0,
        'kelembapan': 5.0
    }
    
    max_change = max_hourly_change.get(param_type, 1.0)
    
    if abs(value - last_value) > max_change:
        if value > last_value:
            value = last_value + max_change
        else:
            value = last_value - max_change
    
    if param_type == 'suhu':
        value = np.clip(value, -30, 60)
    elif param_type == 'kelembapan':
        value = np.clip(value, 0, 100)
    
    return value

@app.on_event("startup")
async def startup_event():
    print("Loading SARIMA models...")
    load_models()
    print("Aplikasi siap!")

@app.get("/")
async def root():
    return {
        "message": "Prediksi Suhu & Kelembapan API v2.0",
        "description": "API untuk prediksi dengan input data manual",
        "status": "running",
        "available_locations": [ "kulkas"],
        "models_loaded": list(models_cache.keys()),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_loaded": len(models_cache),
        "available_models": list(models_cache.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    lokasi = request.lokasi.lower()
    steps = request.steps
    
    valid_locations = ["kulkas"]
    if lokasi not in valid_locations:
        raise HTTPException(
            status_code=400, 
            detail=f"Lokasi tidak valid. Pilih: {valid_locations}"
        )
    
    if steps < 1 or steps > 12:
        raise HTTPException(
            status_code=400,
            detail="Steps harus antara 1-12"
        )
    
    try:
        processed_data, note = process_input_data(request.data_sensor)
        predictions, strategy, is_anomaly, explanation = predict_with_input_data(lokasi, processed_data, steps)
        
        final_note = note
        if final_note and explanation:
            final_note += f" | {explanation}"
        elif explanation:
            final_note = explanation
        
        return PredictionResponse(
            lokasi=lokasi,
            data_points_used=len(processed_data),
            strategy_used=strategy,
            anomaly_detected=is_anomaly,
            note=final_note,
            prediksi=[PredictionItem(**pred) for pred in predictions]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/models")
async def get_available_models():
    model_info = {}
    
    for lokasi, model_data in models_cache.items():
        models = model_data.get('models', {})
        scalers = model_data.get('scalers', {})
        metrics = model_data.get('metrics', {})
        
        model_info[lokasi] = {
            "suhu_model_available": "suhu" in models,
            "kelembapan_model_available": "kelembapan" in models,
            "suhu_scaler_available": "suhu" in scalers,
            "kelembapan_scaler_available": "kelembapan" in scalers
        }
        
        if "suhu" in models:
            try:
                model_info[lokasi]["suhu_aic"] = round(models["suhu"].aic, 2)
            except:
                pass
        
        if "kelembapan" in models:
            try:
                model_info[lokasi]["kelembapan_aic"] = round(models["kelembapan"].aic, 2)
            except:
                pass
        
        if metrics:
            model_info[lokasi]["metrics"] = metrics
    
    return {
        "loaded_models": model_info,
        "total_locations": len(model_info)
    }

@app.get("/example")
async def get_request_example():
    return {
        "example_request": {
            "lokasi": "pantry",
            "steps": 3,
            "data_sensor": [
                {
                    "timestamp": "2025-09-20T10:00:00",
                    "suhu": 25.5,
                    "kelembapan": 65.2
                },
                {
                    "timestamp": "2025-09-20T11:00:00",
                    "suhu": 26.1,
                    "kelembapan": 64.8
                },
                {
                    "timestamp": "2025-09-20T12:00:00",
                    "suhu": 26.8,
                    "kelembapan": 63.5
                }
            ]
        },
        "notes": [
            "Minimal 3 data point, idealnya 12+ untuk akurasi terbaik",
            "Timestamp format ISO (YYYY-MM-DDTHH:MM:SS)",
            "Data otomatis diurutkan berdasarkan timestamp",
            "Sistem otomatis detect anomali dan pilih strategi",
            "Model menggunakan scaled data dengan StandardScaler"
        ]
    }

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", "7860"))  # Spaces default 7860
    uvicorn.run(app, host="0.0.0.0", port=port)