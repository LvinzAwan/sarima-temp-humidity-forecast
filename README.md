# SARIMA-Based Temperature & Humidity Prediction API

This project provides a real-time prediction and risk-scoring API for temperature and humidity data.  
Sensor data is stored in Supabase, processed with a SARIMA time-series model, and served through a FastAPI backend with additional trend analysis and risk scoring.

---

## 1. Project Overview

- **Data source**: `sensor_jam` table in Supabase (filtered by `device_id` and `lokasi`: `ruangan` or `kulkas`).
- **Model**: SARIMA (Seasonal ARIMA) models trained offline for:
  - Temperature (`suhu`)
  - Humidity (`kelembapan`)
- **API**: FastAPI service that:
  - Predicts temperature and humidity **per minute** (up to 120 minutes ahead).
  - Detects anomalies in recent data.
  - Computes a **risk score** and category (from *Sangat Aman* to *Bahaya*).

---

## 2. Metrics Used

Model quality is evaluated with several standard regression metrics:

| Metric | What it measures                                       | How to read it                                  |
| ------ | ------------------------------------------------------ | ----------------------------------------------- |
| **R²** | How well the model follows the pattern in the data     | Closer to 1 → model fits the data better        |
| **MAE**| Average absolute difference between prediction and real data | “On average, how many °C / %RH off”      |
| **RMSE** | Similar to MAE, but more sensitive to large errors   | Penalizes big errors more than small ones       |
| **MAPE** | Average error in percentage                          | “On average, how many percent off from real”    |

These metrics are computed for both training data and a short validation window (e.g., last 180 minutes).

---

## 3. System Architecture

The system has two main parts:

1. **Offline SARIMA Training**
   - Runs locally (e.g., in a training script).
   - Reads historical data from CSV files.
   - Trains and evaluates SARIMA models.
   - Saves trained models as `.pkl` files and uploads them to HuggingFace.

2. **Online FastAPI Service**
   - Loads SARIMA models and scalers from HuggingFace on startup.
   - Fetches recent data from Supabase on each request.
   - Produces per-minute predictions and risk scores via REST endpoints.

High-level data flow:

Supabase → FastAPI → SARIMA + Trend Analysis → Predictions + Risk Scoring → JSON Response

---

## 4. SARIMA Training Pipeline (Offline)

The SARIMA training script follows these steps for both temperature and humidity:

1. **Preprocessing & Resampling**
   - Load CSV with columns: `timestamp`, `suhu`, `kelembapan`.
   - Convert `timestamp` to datetime, sort by time.
   - Resample to **1-minute frequency** and fill missing values with time-based interpolation.

2. **7-Day Sliding Window**
   - Use only the most recent **7 days** of data to:
     - Focus on current behavior.
     - Limit computational cost.
   - Warn if the window contains too few records (e.g., less than 1,000).

3. **Outlier Handling (IQR)**
   - Detect outliers using the Interquartile Range (IQR).
   - Clip extreme values back into a reasonable range.
   - Re-interpolate to keep the time-series smooth.

4. **SARIMA Parameter Search**
   - Test several candidate `(p, d, q)` ARIMA configurations (and seasonal parts if needed).
   - For each configuration:
     - Fit a SARIMAX model on the data.
     - Compute **R²** and **AIC**.
   - Select the configuration with:
     - Highest R² and
     - Lowest AIC (when comparable).
   - Use **early stopping** if R² is already very high (e.g. > 0.95) or if a max search time is exceeded.

5. **Final Training & Evaluation**
   - Retrain the best model with more iterations for better convergence.
   - Compute training and validation metrics:
     - MAE, MSE, RMSE, R², MAPE.
       
       <img width="2712" height="2570" alt="metrics_heatmaps_full" src="https://github.com/user-attachments/assets/4951dc37-ea22-4379-a57a-352b9dbcd11e" />

   - Classify model quality based on R² and error levels (e.g. *SANGAT BAIK*, *BAIK*, *CUKUP*, *PERLU PERBAIKAN*).
   - Save models (temperature + humidity), metrics, and info (window size, frequency, etc.) to a `.pkl` file.

6. **Model Deployment**
   - Upload the `.pkl` model file to a HuggingFace repository.
   - The FastAPI app can then download and use the model at runtime.

---

## 5. FastAPI Service (Online)

The FastAPI app serves two main endpoints:

- `POST /score`  
  Returns **only** the risk score and category (no future predictions).

- `POST /predict`  
  Returns **per-minute predictions** for temperature and humidity along with notes about data and strategy.

### 5.1 Data Preparation (Online)

For each request:

1. Fetch up to **60 latest rows** from Supabase for a given `device_id`.
2. Sort from oldest to newest.
3. Convert `suhu` and `kelembapan` to numeric, interpolate small gaps, and drop invalid rows.
4. Check for physically reasonable ranges:
   - Room: wider limits.
   - Fridge: e.g. temperature in 0–10 °C, humidity in 40–70 %RH.
5. If the number of data points is very small (e.g. < 10 or < 30), attach a note about lower expected accuracy.

### 5.2 Anomaly Detection & Strategy Selection

The function `detect_anomaly_and_choose_strategy` checks:

- Whether min/max values are outside normal ranges (different for **ruangan** and **kulkas**).
- Whether per-minute changes are too large:
  - Temperature change > 0.5 °C/min.
  - Humidity change > 2 %RH/min.

Based on this, it chooses a prediction strategy:

- **`hybrid`**: if data is normal and SARIMA models are available.
- **`trend_based`**: if anomalies are detected or models cannot be used.

### 5.3 Hybrid Prediction (Model + Trend)

In the `hybrid` strategy:

1. **Trend Estimation**
   - Compute a weighted linear trend for the last values (more weight on recent data).
   - Clamp the trend to avoid unrealistic slopes.

2. **Model Forecast**
   - Use the SARIMA models (on scaled data) to forecast the next `steps` minutes.
   - Inverse-scale predictions back to real units (°C and %RH).

3. **Blend Model & Trend**
   - For each prediction step:
     - Combine model and trend with weights.
     - Trend weight is higher if anomalies are present.
     - Trend weight decays slowly over time (damping factor).

4. **Constraints**
   - Limit per-minute changes (e.g., max 0.15 °C/min for temperature).
   - Clip the final predicted values into realistic ranges, e.g.:
     - Fridge temp: 0–10 °C
     - Fridge humidity: 40–70 %RH
     - Room values: broader ranges

### 5.4 Pure Trend Prediction

If the model cannot be used or anomalies are too strong:
- Use only the advanced trend (plus small noise) to generate predictions.
- Apply similar per-minute change limits and value ranges to keep results realistic.

The `/predict` endpoint returns a JSON list:

![WhatsApp Image 2025-11-21 at 13 53 09_fa4b5235](https://github.com/user-attachments/assets/b4b7828d-0cfc-4e57-b998-c0820e77ea9a)

```json
{
  "lokasi": "kulkas",
  "data_points_used": 60,
  "note": "Additional warnings or anomaly notes",
  "prediksi": [
    { "menit_ke": 1, "suhu": 5.9, "kelembapan": 54.3 },
    { "menit_ke": 2, "suhu": 5.8, "kelembapan": 54.1 },
    ...
  ]
}
