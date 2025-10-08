import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class SARIMATrainer:
    def __init__(self, window_days=30, outlier_threshold=3.0):
        self.window_days = window_days
        self.outlier_threshold = outlier_threshold
        self.scalers = {}
        
    def detect_and_remove_outliers(self, series, column_name):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.outlier_threshold * iqr)
        upper_bound = q3 + (self.outlier_threshold * iqr)
        
        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers_mask.sum()
        
        if outlier_count > 0:
            print(f"{column_name}: {outlier_count} outliers detected and corrected")
            series_clean = series.copy()
            series_clean[outliers_mask] = series.rolling(window=5, center=True).median()[outliers_mask]
            series_clean = series_clean.interpolate(method='linear')
            return series_clean
        
        return series
    
    def scale_data(self, data, column, fit=True):
        if fit:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(data[[column]])
            self.scalers[column] = scaler
        else:
            scaler = self.scalers[column]
            scaled_values = scaler.transform(data[[column]])
        
        return pd.Series(scaled_values.flatten(), index=data.index)
    
    def inverse_scale(self, values, column):
        scaler = self.scalers[column]
        return scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    
    def apply_sliding_window(self, df):
        cutoff_date = df.index.max() - timedelta(days=self.window_days)
        df_windowed = df[df.index >= cutoff_date].copy()
        print(f"Sliding window applied: {len(df_windowed)} records from {df_windowed.index.min()} to {df_windowed.index.max()}")
        return df_windowed
    
    def check_stationarity(self, series, name):
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value <= 0.05
        status = "stationary" if is_stationary else "non-stationary"
        print(f"{name}: ADF={result[0]:.4f}, p-value={p_value:.4f} ({status})")
        return is_stationary
    
    def optimize_sarima_params(self, data, column, max_order=2):
        print(f"\nOptimizing SARIMA parameters for {column}...")
        
        p_range = range(0, max_order + 1)
        d_range = range(0, 2)
        q_range = range(0, max_order + 1)
        P_range = range(0, 2)
        D_range = range(0, 2)
        Q_range = range(0, 2)
        s = 24
        
        best_aic = float('inf')
        best_params = None
        best_seasonal = None
        
        configs_tested = 0
        configs_successful = 0
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                configs_tested += 1
                                try:
                                    model = SARIMAX(data[column], 
                                                   order=(p, d, q),
                                                   seasonal_order=(P, D, Q, s),
                                                   enforce_stationarity=False,
                                                   enforce_invertibility=False)
                                    fitted = model.fit(disp=False, maxiter=200)
                                    
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_params = (p, d, q)
                                        best_seasonal = (P, D, Q, s)
                                        print(f"New best: ({p},{d},{q})x({P},{D},{Q},{s}) AIC={best_aic:.2f}")
                                    
                                    configs_successful += 1
                                    
                                except Exception as e:
                                    continue
        
        print(f"Optimization complete: {configs_successful}/{configs_tested} configs successful")
        print(f"Best params: ARIMA{best_params} x {best_seasonal}, AIC={best_aic:.2f}")
        
        return best_params, best_seasonal
    
    def train_model(self, data, column, order, seasonal_order):
        print(f"\nTraining SARIMA model for {column}...")
        print(f"Parameters: ARIMA{order} x {seasonal_order}")
        
        model = SARIMAX(data[column], 
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False, maxiter=200)
        
        print(f"Model trained successfully - AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        
        residuals = fitted_model.resid
        ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//2), return_df=True)
        significant_lags = (ljung_box['lb_pvalue'] < 0.05).sum()
        print(f"Ljung-Box test: {significant_lags}/{len(ljung_box)} lags significant (lower is better)")
        
        return fitted_model
    
    def validate_model(self, model, data, column, test_size=24):
        if len(data) < test_size * 2:
            print("Insufficient data for validation")
            return None
        
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        
        order = model.specification['order']
        seasonal_order = model.specification['seasonal_order']
        
        model_train = SARIMAX(train_data[column],
                             order=order,
                             seasonal_order=seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        
        fitted_train = model_train.fit(disp=False, maxiter=200)
        predictions = fitted_train.forecast(steps=test_size)
        
        actual_scaled = test_data[column].values
        pred_scaled = predictions.values
        
        actual_original = self.inverse_scale(actual_scaled, column)
        pred_original = self.inverse_scale(pred_scaled, column)
        
        mae = np.mean(np.abs(actual_original - pred_original))
        rmse = np.sqrt(np.mean((actual_original - pred_original) ** 2))
        mape = np.mean(np.abs((actual_original - pred_original) / actual_original)) * 100
        
        print(f"Validation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}

def process_location(lokasi, trainer, optimize=False):
    print(f"\n{'='*60}")
    print(f"PROCESSING: {lokasi.upper()}")
    print(f"{'='*60}")
    
    csv_file = f"data/{lokasi}.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded: {df.shape[0]} records")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        return None
    
    required_cols = ['timestamp', 'suhu', 'kelembapan']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    df['suhu'] = df['suhu'].interpolate(method='linear')
    df['kelembapan'] = df['kelembapan'].interpolate(method='linear')
    
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    
    df = trainer.apply_sliding_window(df)
    
    print("\nCleaning outliers...")
    df['suhu'] = trainer.detect_and_remove_outliers(df['suhu'], 'suhu')
    df['kelembapan'] = trainer.detect_and_remove_outliers(df['kelembapan'], 'kelembapan')
    
    df_scaled = df.copy()
    df_scaled['suhu'] = trainer.scale_data(df, 'suhu', fit=True)
    df_scaled['kelembapan'] = trainer.scale_data(df, 'kelembapan', fit=True)
    
    print("\nStationarity check (scaled data):")
    trainer.check_stationarity(df_scaled['suhu'], 'suhu')
    trainer.check_stationarity(df_scaled['kelembapan'], 'kelembapan')
    
    models = {}
    metrics = {}
    
    for var in ['suhu', 'kelembapan']:
        print(f"\n{'-'*40}")
        print(f"TRAINING: {var.upper()}")
        print(f"{'-'*40}")
        
        if optimize:
            order, seasonal = trainer.optimize_sarima_params(df_scaled, var)
        else:
            order = (1, 1, 1)
            seasonal = (1, 1, 1, 24)
            print(f"Using default parameters: ARIMA{order} x {seasonal}")
        
        model = trainer.train_model(df_scaled, var, order, seasonal)
        models[var] = model
        
        validation_metrics = trainer.validate_model(model, df_scaled, var)
        if validation_metrics:
            metrics[var] = validation_metrics
    
    model_data = {
        'models': models,
        'scalers': trainer.scalers.copy(),
        'metrics': metrics,
        'last_training': datetime.now(),
        'window_days': trainer.window_days,
        'data_range': (df.index.min(), df.index.max())
    }
    
    model_file = f"models/{lokasi}_models.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {model_file}")
    
    print("\nTesting forecast (3 hours ahead):")
    for var in ['suhu', 'kelembapan']:
        forecast_scaled = models[var].forecast(steps=3)
        forecast_original = trainer.inverse_scale(forecast_scaled.values, var)
        
        print(f"\n{var.capitalize()}:")
        for i, val in enumerate(forecast_original, 1):
            print(f"  Hour {i}: {val:.2f}")
    
    return model_data

def main():
    os.makedirs('models', exist_ok=True)
    
    print("="*60)
    print("SARIMA TRAINING SYSTEM")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = SARIMATrainer(window_days=30, outlier_threshold=3.0)
    
    locations = ['kulkas']
    optimize_params = False
    
    results = {}
    for lokasi in locations:
        result = process_location(lokasi, trainer, optimize=optimize_params)
        if result:
            results[lokasi] = result
        
        trainer.scalers = {}
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    for lokasi in locations:
        model_file = f"models/{lokasi}_models.pkl"
        if os.path.exists(model_file):
            print(f" V {model_file}")
            if lokasi in results and 'metrics' in results[lokasi]:
                for var, m in results[lokasi]['metrics'].items():
                    print(f"  {var}: MAPE={m['mape']:.2f}%, RMSE={m['rmse']:.2f}")
        else:
            print(f" X {model_file}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()