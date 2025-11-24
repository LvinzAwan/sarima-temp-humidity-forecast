from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
import numpy as np
from typing import Optional, List, Dict
import warnings
import requests
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Prediksi Suhu & Kelembapan API",
    description="API untuk prediksi suhu dan kelembapan menggunakan model SARIMA"
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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://rcvbwyvnnuurudizkuec.supabase.co")
SUPABASE_SCHEMA = os.environ.get("SUPABASE_SCHEMA", "rpl")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "sensor_jam")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJjdmJ3eXZubnV1cnVkaXprdWVjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgxOTgzNDksImV4cCI6MjA3Mzc3NDM0OX0.YCXPxFi7IQvWKN16soE0-YA_mziN9uN2B1wYkOfuhrc")

SUPABASE_BASE_URL = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

class ApiRequest(BaseModel):
    lokasi: str = Field(..., description="Lokasi sensor: ruangan atau kulkas")
    device_id: int = Field(..., description="ID device di Supabase")
    steps: Optional[int] = Field(default=60, description="Jumlah menit prediksi (1-120)")

class PredictionItem(BaseModel):
    menit_ke: int
    suhu: float
    kelembapan: float

class RiskScoreDetail(BaseModel):
    skor_dasar: float
    skor_trend: float
    skor_stabilitas: float

class RiskScore(BaseModel):
    persentase: float
    kategori: str
    detail: RiskScoreDetail

class ScoreResponse(BaseModel):
    lokasi: str
    data_points_used: int
    anomaly_detected: bool
    strategy_suggested: str
    skoring: RiskScore
    note: Optional[str] = None

class PredictionResponse(BaseModel):
    lokasi: str
    data_points_used: int
    note: Optional[str] = None
    prediksi: List[PredictionItem]

models_cache = {}
model_stats = {}

def fetch_data_from_supabase(device_id: int) -> pd.DataFrame:
    url = f"{SUPABASE_BASE_URL}?select=*&device_id=eq.{device_id}&order=timestamp.desc&limit=60"
    
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Accept": "application/json",
        "Accept-Profile": SUPABASE_SCHEMA,
        "Content-Profile": SUPABASE_SCHEMA
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail=f"Tidak ada data untuk device_id {device_id}")

        df = pd.DataFrame(data)
        df = df.rename(columns={'temp': 'suhu', 'humidity': 'kelembapan'})

        if 'suhu' not in df.columns or 'kelembapan' not in df.columns:
            raise HTTPException(status_code=500, detail="Kolom suhu atau kelembapan tidak ditemukan")

        df = df[['timestamp', 'suhu', 'kelembapan']]
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            pass

        df = df.sort_values('timestamp', na_position='first').reset_index(drop=True)
        return df

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

def prepare_data_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) > 60:
        df = df.tail(60).reset_index(drop=True)

    df['suhu'] = pd.to_numeric(df['suhu'], errors='coerce')
    df['kelembapan'] = pd.to_numeric(df['kelembapan'], errors='coerce')
    df['suhu'] = df['suhu'].interpolate()
    df['kelembapan'] = df['kelembapan'].interpolate()
    df = df.dropna()

    return df

def common_prepare(lokasi: str, device_id: int, steps: int) -> tuple[pd.DataFrame, Optional[str]]:
    valid_locations = ["ruangan", "kulkas"]
    if lokasi.lower() not in valid_locations:
        raise HTTPException(status_code=400, detail=f"Lokasi tidak valid. Pilih: {valid_locations}")
    
    if steps < 1 or steps > 120:
        raise HTTPException(status_code=400, detail="Steps harus antara 1-120 menit")
    
    df = fetch_data_from_supabase(device_id)
    df = prepare_data_per_minute(df)
    
    note = None
    data_count = len(df)
    
    if data_count < 10:
        note = f"Data hanya {data_count} menit, akurasi mungkin kurang optimal"
    elif data_count < 30:
        note = f"Data hanya {data_count} menit, untuk hasil terbaik gunakan 30+ data point"
    
    df['suhu'] = pd.to_numeric(df['suhu'], errors='coerce')
    df['kelembapan'] = pd.to_numeric(df['kelembapan'], errors='coerce')
    
    if df['suhu'].isna().any() or df['kelembapan'].isna().any():
        df['suhu'] = df['suhu'].interpolate(method='linear')
        df['kelembapan'] = df['kelembapan'].interpolate(method='linear')
        
        if df['suhu'].isna().any() or df['kelembapan'].isna().any():
            raise HTTPException(status_code=400, detail="Data mengandung nilai yang tidak valid")
    
    if df['suhu'].min() < -50 or df['suhu'].max() > 100:
        note = (note or "") + " Nilai suhu di luar rentang normal"
    
    if df['kelembapan'].min() < 0 or df['kelembapan'].max() > 100:
        note = (note or "") + " Nilai kelembapan di luar rentang normal"
    
    return df, note

def download_model_from_hf(lokasi: str):
    model_repo = "pauluswindi/prediction-sarima-models"
    model_url = f"https://huggingface.co/{model_repo}/resolve/main/{lokasi}_model.pkl"
    model_path = f"/tmp/{lokasi}_model.pkl"
    
    try:
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        return model_path
    except Exception as e:
        print(f"Error downloading model {lokasi}: {str(e)}")
        return None

def load_models():
    global models_cache, model_stats
    lokasi_list = ['ruangan', 'kulkas']
    
    for lokasi in lokasi_list:
        model_file = f"/tmp/{lokasi}_model.pkl"
        
        if not os.path.exists(model_file):
            model_file = download_model_from_hf(lokasi)
        
        if model_file and os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                if isinstance(loaded_data, dict) and 'models' in loaded_data:
                    actual_models = loaded_data['models']
                    
                    if isinstance(actual_models, dict) and 'suhu' in actual_models and 'kelembapan' in actual_models:
                        models_cache[lokasi] = {
                            'models': actual_models,
                            'metrics': loaded_data.get('metrics', {}),
                            'frequency': loaded_data.get('frequency', 'per_minute')
                        }
                        model_stats[lokasi] = get_default_model_stats(lokasi)
                        
            except Exception as e:
                print(f"Error loading model {lokasi}: {str(e)}")

def get_default_model_stats(lokasi: str) -> dict:
    stats_mapping = {
        'kulkas': {
            'suhu': {'mean': 6.0, 'std': 2.0, 'min': 0.0, 'max': 10.0},
            'kelembapan': {'mean': 55.0, 'std': 5.0, 'min': 40.0, 'max': 70.0}
        },
        'ruangan': {
            'suhu': {'mean': 25.0, 'std': 3.0, 'min': 18.0, 'max': 32.0},
            'kelembapan': {'mean': 60.0, 'std': 10.0, 'min': 40.0, 'max': 80.0}
        }
    }
    return stats_mapping.get(lokasi, stats_mapping['ruangan'])

def calculate_risk_score(data: pd.DataFrame, lokasi: str) -> dict:
    suhu_data = data['suhu']
    kelembapan_data = data['kelembapan']
    suhu_terakhir = suhu_data.iloc[-1]
    kelembapan_terakhir = kelembapan_data.iloc[-1]

    if lokasi.lower() == 'kulkas':
        ideal_suhu = 6.0
        ideal_kelembapan = 55.0
        suhu_tolerance = 4.0
        kelembapan_tolerance = 15.0
    else:
        ideal_suhu = 25.0
        ideal_kelembapan = 60.0
        suhu_tolerance = 5.0
        kelembapan_tolerance = 15.0

    deviasi_suhu = abs(suhu_terakhir - ideal_suhu)
    deviasi_kelembapan = abs(kelembapan_terakhir - ideal_kelembapan)

    skor_suhu = max(0, 100 - (deviasi_suhu / suhu_tolerance) * 100)
    skor_kelembapan = max(0, 100 - (deviasi_kelembapan / kelembapan_tolerance) * 100)
    skor_dasar = (0.6 * skor_suhu) + (0.4 * skor_kelembapan)

    trend_suhu = suhu_data.iloc[-1] - suhu_data.iloc[0]
    trend_kelembapan = kelembapan_data.iloc[-1] - kelembapan_data.iloc[0]
    penalti_trend = 0

    if lokasi.lower() == 'kulkas':
        if trend_suhu > 0.5:
            penalti_trend += 10
        elif trend_suhu < -0.5:
            penalti_trend -= 5
        if trend_kelembapan < -5:
            penalti_trend += 10
        elif trend_kelembapan > 0 and kelembapan_terakhir <= 70:
            penalti_trend -= 5
    else:
        if abs(trend_suhu) > 1.0:
            penalti_trend += 10
        if abs(trend_kelembapan) > 5:
            penalti_trend += 10

    skor_trend = max(0, min(100, 100 - penalti_trend))

    std_suhu = np.std(suhu_data)
    std_kelembapan = np.std(kelembapan_data)

    if lokasi.lower() == 'kulkas':
        stabilitas_suhu = max(0, 100 - (std_suhu / 1.5) * 100)
        stabilitas_kelembapan = max(0, 100 - (std_kelembapan / 5.0) * 100)
    else:
        stabilitas_suhu = max(0, 100 - (std_suhu / 2.0) * 100)
        stabilitas_kelembapan = max(0, 100 - (std_kelembapan / 5.0) * 100)

    skor_stabilitas = (stabilitas_suhu + stabilitas_kelembapan) / 2
    skor_akhir = (0.5 * skor_dasar) + (0.3 * skor_trend) + (0.2 * skor_stabilitas)

    if skor_akhir >= 90:
        kategori = "Sangat Aman"
    elif skor_akhir >= 70:
        kategori = "Aman"
    elif skor_akhir >= 55:
        kategori = "Waspada"
    elif skor_akhir >= 35:
        kategori = "Berisiko"
    else:
        kategori = "Bahaya"

    return {
        "persentase": round(skor_akhir, 2),
        "kategori": kategori,
        "detail": {
            "skor_dasar": round(skor_dasar, 2),
            "skor_trend": round(skor_trend, 2),
            "skor_stabilitas": round(skor_stabilitas, 2)
        }
    }

def detect_anomaly_and_choose_strategy(data: pd.DataFrame, lokasi: str) -> tuple[str, bool, str]:
    stats = model_stats.get(lokasi, {})
    suhu_stats = stats.get('suhu', {})
    kelembapan_stats = stats.get('kelembapan', {})

    current_suhu_min = data['suhu'].min()
    current_suhu_max = data['suhu'].max()
    current_kelembapan_min = data['kelembapan'].min()
    current_kelembapan_max = data['kelembapan'].max()

    anomaly_reasons = []

    if lokasi.lower() == 'kulkas':
        suhu_lower, suhu_upper = 0.0, 10.0
    else:
        suhu_lower = suhu_stats.get('mean', 25) - 3 * suhu_stats.get('std', 3)
        suhu_upper = suhu_stats.get('mean', 25) + 3 * suhu_stats.get('std', 3)

    if current_suhu_min < suhu_lower or current_suhu_max > suhu_upper:
        anomaly_reasons.append(f"Suhu di luar rentang normal ({suhu_lower:.1f}°C - {suhu_upper:.1f}°C)")

    if lokasi.lower() == 'kulkas':
        kelembapan_lower, kelembapan_upper = 40.0, 70.0
    else:
        kelembapan_lower = kelembapan_stats.get('mean', 60) - 3 * kelembapan_stats.get('std', 10)
        kelembapan_upper = kelembapan_stats.get('mean', 60) + 3 * kelembapan_stats.get('std', 10)

    if current_kelembapan_min < kelembapan_lower or current_kelembapan_max > kelembapan_upper:
        anomaly_reasons.append(f"Kelembapan di luar rentang normal ({kelembapan_lower:.1f}% - {kelembapan_upper:.1f}%)")

    if len(data) >= 2:
        suhu_changes = abs(data['suhu'].diff().dropna())
        kelembapan_changes = abs(data['kelembapan'].diff().dropna())

        if suhu_changes.max() > 0.5:
            anomaly_reasons.append(f"Perubahan suhu drastis: {suhu_changes.max():.2f}°C/menit")
        if kelembapan_changes.max() > 2.0:
            anomaly_reasons.append(f"Perubahan kelembapan drastis: {kelembapan_changes.max():.2f}%/menit")

    is_anomaly = len(anomaly_reasons) > 0
    strategy = "trend_based" if is_anomaly else "hybrid"
    explanation = "; ".join(anomaly_reasons) if is_anomaly else "Kondisi normal"

    return strategy, is_anomaly, explanation

def predict_with_input_data(lokasi: str, data: pd.DataFrame, steps: int = 60) -> tuple[List[dict], str, bool, str]:
    strategy, is_anomaly, explanation = detect_anomaly_and_choose_strategy(data, lokasi)
    
    if lokasi not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model untuk lokasi '{lokasi}' tidak tersedia")
    
    model_data = models_cache[lokasi]
    models = model_data['models']
    
    if not isinstance(models, dict):
        raise HTTPException(status_code=500, detail=f"Format model untuk lokasi '{lokasi}' tidak valid")
    
    if 'suhu' not in models or 'kelembapan' not in models:
        raise HTTPException(status_code=500, detail=f"Model tidak lengkap untuk lokasi '{lokasi}'")

    if strategy == "hybrid":
        try:
            if not hasattr(models['suhu'], 'forecast') or not hasattr(models['kelembapan'], 'forecast'):
                strategy = "trend_based"
                explanation += " | Model tidak support forecast"
        except Exception as e:
            strategy = "trend_based"
            explanation += f" | Error: {str(e)}"
    
    try:
        data_indexed = data.set_index('timestamp')
        suhu_data = data_indexed['suhu'].dropna()
        kelembapan_data = data_indexed['kelembapan'].dropna()
        
        if strategy == "trend_based":
            predictions = predict_pure_trend(suhu_data, kelembapan_data, steps, lokasi)
        else:
            try:
                trend_weight = 0.6 if is_anomaly else 0.3
                predictions = predict_hybrid(suhu_data, kelembapan_data, models['suhu'], models['kelembapan'], steps, trend_weight, lokasi)
            except Exception:
                predictions = predict_pure_trend(suhu_data, kelembapan_data, steps, lokasi)
                strategy = "trend_based"
                explanation = "Fallback ke trend-based"
        
        return predictions, strategy, is_anomaly, explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")

def predict_hybrid(suhu_data: pd.Series, kelembapan_data: pd.Series, suhu_model, kelembapan_model, steps: int, trend_weight: float, lokasi: str) -> List[dict]:
    suhu_trend = calculate_trend(suhu_data.values)
    kelembapan_trend = calculate_trend(kelembapan_data.values)
    
    last_suhu = suhu_data.iloc[-1]
    last_kelembapan = kelembapan_data.iloc[-1]
    
    model_suhu_forecast = suhu_model.forecast(steps=steps)
    model_kelembapan_forecast = kelembapan_model.forecast(steps=steps)
    
    predictions = []
    
    for i in range(steps):
        current_trend_weight = trend_weight * (0.95 ** i)
        current_model_weight = 1 - current_trend_weight
        
        trend_suhu = last_suhu + (suhu_trend * (i + 1))
        trend_kelembapan = last_kelembapan + (kelembapan_trend * (i + 1))
        
        final_suhu = (trend_suhu * current_trend_weight) + (float(model_suhu_forecast.iloc[i]) * current_model_weight)
        final_kelembapan = (trend_kelembapan * current_trend_weight) + (float(model_kelembapan_forecast.iloc[i]) * current_model_weight)
        
        final_suhu = apply_constraints(final_suhu, 'suhu', last_suhu, lokasi)
        final_kelembapan = apply_constraints(final_kelembapan, 'kelembapan', last_kelembapan, lokasi)
        
        predictions.append({
            "menit_ke": i + 1,
            "suhu": round(final_suhu, 2),
            "kelembapan": round(final_kelembapan, 2)
        })
    
    return predictions

def predict_pure_trend(suhu_data: pd.Series, kelembapan_data: pd.Series, steps: int, lokasi: str) -> List[dict]:
    suhu_trend = calculate_trend(suhu_data.values)
    kelembapan_trend = calculate_trend(kelembapan_data.values)
    
    last_suhu = suhu_data.iloc[-1]
    last_kelembapan = kelembapan_data.iloc[-1]
    
    suhu_volatility = np.std(np.diff(suhu_data.values))
    kelembapan_volatility = np.std(np.diff(kelembapan_data.values))
    
    predictions = []
    
    for i in range(steps):
        damping_factor = 0.95 ** i
        
        pred_suhu = last_suhu + (suhu_trend * (i + 1) * damping_factor)
        pred_kelembapan = last_kelembapan + (kelembapan_trend * (i + 1) * damping_factor)
        
        pred_suhu += np.random.normal(0, suhu_volatility * 0.1)
        pred_kelembapan += np.random.normal(0, kelembapan_volatility * 0.1)
        
        pred_suhu = apply_constraints(pred_suhu, 'suhu', last_suhu, lokasi, i + 1)
        pred_kelembapan = apply_constraints(pred_kelembapan, 'kelembapan', last_kelembapan, lokasi, i + 1)
        
        predictions.append({
            "menit_ke": i + 1,
            "suhu": round(pred_suhu, 2),
            "kelembapan": round(pred_kelembapan, 2)
        })
    
    return predictions

def calculate_trend(values: np.ndarray) -> float:
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

def apply_constraints(value: float, param_type: str, last_value: float, lokasi: str, step: int = 1) -> float:
    max_change_per_minute = {'suhu': 0.15, 'kelembapan': 0.5}
    max_change = max_change_per_minute.get(param_type, 0.1) * step

    if abs(value - last_value) > max_change:
        value = last_value + max_change if value > last_value else last_value - max_change

    if param_type == 'suhu':
        if lokasi == "kulkas":
            value = np.clip(value, 0, 10)
        else:
            value = np.clip(value, -30, 60)
    elif param_type == 'kelembapan':
        if lokasi == "kulkas":
            value = np.clip(value, 40, 70)
        else:
            value = np.clip(value, 0, 100)

    return value

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Prediksi Suhu & Kelembapan API",
        "status": "running",
        "available_locations": ["ruangan", "kulkas"],
        "models_loaded": list(models_cache.keys()),
        "endpoints": {
            "/score": "Risk scoring only",
            "/predict": "Full predictions with risk scoring"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models_cache),
        "available_models": list(models_cache.keys())
    }

@app.post("/score", response_model=ScoreResponse)
async def score_endpoint(request: ApiRequest):
    lokasi = request.lokasi.lower()
    
    try:
        processed_data, note = common_prepare(lokasi, request.device_id, request.steps)
        risk_score = calculate_risk_score(processed_data, lokasi)
        strategy, is_anomaly, explanation = detect_anomaly_and_choose_strategy(processed_data, lokasi)
        
        final_note = note
        if final_note and explanation:
            final_note += f" | {explanation}"
        elif explanation:
            final_note = explanation
        
        return ScoreResponse(
            lokasi=lokasi,
            data_points_used=len(processed_data),
            anomaly_detected=is_anomaly,
            strategy_suggested=strategy,
            skoring=RiskScore(
                persentase=risk_score["persentase"],
                kategori=risk_score["kategori"],
                detail=RiskScoreDetail(**risk_score["detail"])
            ),
            note=final_note
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: ApiRequest):
    lokasi = request.lokasi.lower()
    steps = request.steps
    
    try:
        processed_data, note = common_prepare(lokasi, request.device_id, steps)
        predictions, strategy, is_anomaly, explanation = predict_with_input_data(lokasi, processed_data, steps)
        
        final_note = note
        if final_note and explanation:
            final_note += f" | {explanation}"
        elif explanation:
            final_note = explanation
        
        return PredictionResponse(
            lokasi=lokasi,
            data_points_used=len(processed_data),
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
        metrics = model_data.get('metrics', {})
        frequency = model_data.get('frequency', 'unknown')
        
        model_info[lokasi] = {
            "suhu_model_available": "suhu" in models,
            "kelembapan_model_available": "kelembapan" in models,
            "frequency": frequency
        }
        
        if metrics:
            model_info[lokasi]["metrics"] = metrics
    
    return {"loaded_models": model_info, "total_locations": len(model_info)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
