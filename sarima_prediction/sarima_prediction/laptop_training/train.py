import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import glob
import warnings
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SARIMATrainer:
    def __init__(self, window_days=7, outlier_threshold=3.0):
        self.window_days = window_days
        self.outlier_threshold = outlier_threshold

    # =================== PREPROCESSING ===================

    def detect_and_remove_outliers(self, series, column_name):
        """Deteksi dan koreksi outlier menggunakan IQR clipping"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.outlier_threshold * iqr)
        upper_bound = q3 + (self.outlier_threshold * iqr)

        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers_mask.sum()

        if outlier_count > 0:
            print(
                f"{column_name}: Terdeteksi {outlier_count} outlier, "
                f"dikoreksi ke rentang [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
            series_clipped = series.clip(lower=lower_bound, upper=upper_bound)
            series_clean = series_clipped.interpolate(method='time')
            return series_clean

        return series

    def apply_sliding_window(self, df):
        """Pilih data terbaru dalam window hari"""
        cutoff_date = df.index.max() - timedelta(days=self.window_days)
        df_windowed = df[df.index >= cutoff_date].copy()

        if len(df_windowed) < 1000:
            print(
                f"PERINGATAN: Data terlalu sedikit ({len(df_windowed)} records), "
                f"direkomendasikan minimal 1000"
            )

        print(
            f"Sliding window: {len(df_windowed)} records dari "
            f"{df_windowed.index.min()} to {df_windowed.index.max()}"
        )
        return df_windowed

    def check_stationarity(self, series, name):
        """Cek stasioneritas data"""
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value <= 0.05
        status = "stasioner" if is_stationary else "non-stasioner (perlu differencing)"
        print(f"{name}: ADF Statistic={result[0]:.4f}, p-value={p_value:.6f} ({status})")
        return is_stationary

    # =================== MODELING ===================

    def optimize_sarima_params(self, data, column, max_search_time=180):
        """Optimasi parameter dengan timeout dan early stopping"""
        print(f"\nMencari parameter optimal untuk {column}...")

        if len(data) > 5000:
            configs = [
                ((1, 0, 1), (0, 0, 0, 0)),
                ((2, 0, 1), (0, 0, 0, 0)),
                ((1, 1, 1), (0, 0, 0, 0)),
            ]
        else:
            configs = [
                ((1, 0, 0), (0, 0, 0, 0)),
                ((1, 0, 1), (0, 0, 0, 0)),
                ((2, 0, 1), (0, 0, 0, 0)),
                ((0, 1, 1), (0, 0, 0, 0)),
                ((1, 1, 1), (0, 0, 0, 0)),
            ]

        best_r2 = -np.inf
        best_aic = np.inf
        best_params = None
        best_seasonal = None

        tested = 0
        successful = 0
        start_time = time.time()

        print(f"Testing {len(configs)} konfigurasi (max {max_search_time}s)...")

        for order, seasonal in configs:
            elapsed = time.time() - start_time
            if elapsed > max_search_time:
                print(
                    f"\n⏱ Timeout {max_search_time}s tercapai, "
                    f"gunakan model terbaik saat ini"
                )
                break

            # skip seasonal period yang tidak masuk akal
            if seasonal[3] > len(data) // 2:
                continue

            tested += 1
            try:
                print(f"\n  Testing {order}x{seasonal}... ", end="", flush=True)
                config_start = time.time()

                model = SARIMAX(
                    data[column],
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=True
                )

                fitted = model.fit(disp=False, maxiter=50, method='lbfgs')

                fitted_vals = fitted.fittedvalues
                actual_vals = data[column].values[-len(fitted_vals):]

                ss_res = np.sum((actual_vals - fitted_vals) ** 2)
                ss_tot = np.sum(
                    (actual_vals - np.mean(actual_vals)) ** 2
                )
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf

                config_time = time.time() - config_start
                print(
                    f"R²={r_squared:.4f} | AIC={fitted.aic:.2f} | "
                    f"{config_time:.1f}s"
                )

                if r_squared > best_r2:
                    best_r2 = r_squared
                    best_aic = fitted.aic
                    best_params = order
                    best_seasonal = seasonal

                    if r_squared > 0.95:
                        print("  ✓ R² > 0.95, early stopping!")
                        successful += 1
                        break

                successful += 1

            except Exception as error:
                print(f"✗ Gagal: {str(error)[:50]}")
                continue

        print(
            f"\nHasil: {successful}/{tested} konfigurasi berhasil "
            f"dalam {time.time()-start_time:.1f}s"
        )
        print(f"Parameter TERBAIK: ARIMA{best_params} x {best_seasonal}")
        print(f"  R² = {best_r2:.4f} | AIC = {best_aic:.2f}")

        return best_params, best_seasonal

    def train_model(self, data, column, order, seasonal_order):
        """Latih model final dengan iterasi lebih banyak untuk konvergensi"""
        print(f"\nMelatih model SARIMA untuk {column}...")
        print(f"Parameter: ARIMA{order} x {seasonal_order}")

        try:
            model = SARIMAX(
                data[column],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True
            )

            fitted_model = model.fit(disp=False, maxiter=200, method='lbfgs')

            print(
                f"✓ Model berhasil dilatih - AIC: {fitted_model.aic:.2f}, "
                f"BIC: {fitted_model.bic:.2f}"
            )
            return fitted_model

        except Exception as error:
            print(f"✗ Training gagal: {error}")
            return None

    # =================== SCORING & METRICS ===================

    def _score_abs_error(self, variable, value):
        """
        Kategori MAE/RMSE berbeda untuk suhu & kelembapan.

        SUHU (°C):
          <0.5   -> SANGAT BAIK
          0.5–1  -> BAIK
          1–2    -> CUKUP
          >=2    -> PERLU PERBAIKAN

        KELEMBAPAN (%RH):
          <2     -> SANGAT BAIK
          2–5    -> BAIK
          5–10   -> CUKUP
          >=10   -> PERLU PERBAIKAN
        """
        if variable is None:
            # fallback generik
            if value < 5:
                return "SANGAT BAIK"
            elif value < 10:
                return "BAIK"
            elif value < 20:
                return "CUKUP"
            else:
                return "PERLU PERBAIKAN"

        v = variable.lower()

        if "suhu" in v:
            if value < 0.5:
                return "SANGAT BAIK"
            elif value < 1.0:
                return "BAIK"
            elif value < 2.0:
                return "CUKUP"
            else:
                return "PERLU PERBAIKAN"

        if "kelembapan" in v:
            if value < 2.0:
                return "SANGAT BAIK"
            elif value < 5.0:
                return "BAIK"
            elif value < 10.0:
                return "CUKUP"
            else:
                return "PERLU PERBAIKAN"

        # kalau nama variabel lain
        if value < 5:
            return "SANGAT BAIK"
        elif value < 10:
            return "BAIK"
        elif value < 20:
            return "CUKUP"
        else:
            return "PERLU PERBAIKAN"

    def calculate_metrics(self, actual, predicted, variable=None):
        """Hitung metrik (absolute + MAPE) dengan kategori MAE/RMSE per variabel"""
        n = min(len(actual), len(predicted))
        actual = np.asarray(actual[-n:])
        predicted = np.asarray(predicted[-n:])

        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf

        # error persen terhadap nilai aktual (MAPE)
        mape = (
            np.mean(np.abs((actual - predicted) / actual)) * 100
            if np.mean(np.abs(actual)) > 0
            else np.inf
        )

        performance = (
            "SANGAT BAIK" if r_squared >= 0.9 else
            "BAIK" if r_squared >= 0.7 else
            "CUKUP" if r_squared >= 0.5 else
            "PERLU PERBAIKAN"
        )

        mae_score = self._score_abs_error(variable, mae)
        rmse_score = self._score_abs_error(variable, rmse)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r_squared': r_squared,
            'mape': mape,
            'performance': performance,
            'mean': np.mean(actual),
            'std': np.std(actual),
            'mae_score': mae_score,
            'rmse_score': rmse_score
        }

    def display_metrics(self, metrics, variable, data_type="Training"):
        """Tampilkan metrik dalam format tabel"""
        print(f"\n{'='*70}")
        print(f"{variable.upper()} - METRIK EVALUASI ({data_type.upper()} DATA)")
        print(f"{'='*70}")
        print(f"  MAE (Mean Absolute Error)       : {metrics['mae']:>8.4f}  "
              f"({metrics['mae_score']})")
        print(f"  MSE (Mean Squared Error)        : {metrics['mse']:>8.4f}")
        print(f"  RMSE (Root Mean Squared Error)  : {metrics['rmse']:>8.4f}  "
              f"({metrics['rmse_score']})")
        print(
            f"  R²  (R-squared)                 : {metrics['r_squared']:>8.4f}  "
            f"({metrics['r_squared']*100:>6.2f}%)"
        )
        print(f"  MAPE (Mean Absolute % Error)    : {metrics['mape']:>8.2f}%")
        print(f"\n  Performa Model (berdasarkan R²): {metrics['performance']}")
        print(
            f"\n  Statistik Data: Mean={metrics['mean']:.2f}, "
            f"Std={metrics['std']:.2f}"
        )

    # =================== VISUALISASI ===================

    def plot_model_comparison_matrix(self, models_dict, data, save_dir):
        """
        Plot:
        - Tabel metrik utama (file terpisah)
        - Satu gambar berisi:
            1) R² heatmap (0–1)
            2) Absolute error (MAE & RMSE)
            3) Percentage error (MAPE)
        """

        if len(models_dict) < 2:
            print("  ⚠ Matriks komparasi membutuhkan minimal 2 model")
            return

        # Siapkan data untuk matriks
        metrics_data = []
        for var, model_info in models_dict.items():
            if model_info['model'] is not None:
                metrics_data.append({
                    'Variable': var.upper(),
                    'Model': f"ARIMA{model_info['order']}x{model_info['seasonal']}",
                    'R²': model_info['metrics']['r_squared'],
                    'MAE': model_info['metrics']['mae'],
                    'RMSE': model_info['rmse'],
                    'MAPE (%)': model_info['metrics']['mape'],
                    'AIC': model_info['model'].aic,
                    'Performance': model_info['metrics']['performance']
                })

        if not metrics_data:
            return

        # ---------- 1. TABEL METRIK UTAMA ----------
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axis('off')

        table_vals = []
        for row in metrics_data:
            table_vals.append([
                row['Variable'],
                row['Model'],
                f"{row['R²']:.4f}",
                f"{row['MAE']:.4f}",
                f"{row['RMSE']:.4f}",
                f"{row['MAPE (%)']:.2f}%",
                f"{row['AIC']:.0f}",
                row['Performance']
            ])

        headers = [
            'Variable', 'Model', 'R² Score',
            'MAE', 'RMSE',
            'MAPE (%)', 'AIC', 'Performance'
        ]

        table = ax.table(
            cellText=table_vals,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2C3E50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        performance_colors = {
            'SANGAT BAIK': '#27AE60',
            'BAIK': '#F39C12',
            'CUKUP': '#3498DB',
            'PERLU PERBAIKAN': '#E74C3C'
        }

        for i, row in enumerate(metrics_data):
            perf = row['Performance']
            table[(i + 1, 7)].set_facecolor(performance_colors.get(perf, 'white'))
            table[(i + 1, 7)].set_text_props(weight='bold')

        ax.set_title(
            'MODEL COMPARISON MATRIX - PERFORMANCE METRICS',
            fontsize=16, fontweight='bold', pad=20
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'model_comparison_matrix.png'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
        plt.close()

        # ---------- 2. GABUNG R² + ERROR DALAM 1 GAMBAR ----------
        index_labels = [row['Variable'] for row in metrics_data]

        r2_data = pd.DataFrame(
            {'R²': [row['R²'] for row in metrics_data]},
            index=index_labels
        )

        error_abs = pd.DataFrame({
            'MAE': [row['MAE'] for row in metrics_data],
            'RMSE': [row['RMSE'] for row in metrics_data]
        }, index=index_labels)

        error_pct = pd.DataFrame({
            'MAPE (%)': [row['MAPE (%)'] for row in metrics_data],
        }, index=index_labels)

        fig, axes = plt.subplots(
            3, 1, figsize=(12, 9),
            gridspec_kw={'height_ratios': [0.9, 1, 1], 'hspace': 0.35}
        )
        ax_r2, ax_abs, ax_pct = axes

        # 2a. R² heatmap (0.0–1.0)
        sns.heatmap(
            r2_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'R² (0–1, lebih tinggi lebih baik)'},
            ax=ax_r2,
            linewidths=1,
            linecolor='white',
            vmin=0.0,
            vmax=1.0
        )
        ax_r2.set_title(
            'R² Score Comparison (Higher is Better)',
            fontsize=13, fontweight='bold'
        )
        ax_r2.set_ylabel('Variables', fontsize=11)

        # 2b. Absolute error (MAE & RMSE)
        # Range warna global, tapi kategori MAE/RMSE sudah beda per variabel
        abs_max = float(error_abs.values.max())
        abs_vmax = max(5.0, abs_max)  # supaya nilai kecil tetap hijau

        sns.heatmap(
            error_abs,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',   # hijau = kecil, merah = besar
            cbar_kws={'label': 'Absolute Error (lebih kecil lebih baik)'},
            ax=ax_abs,
            linewidths=1,
            linecolor='white',
            vmin=0.0,
            vmax=abs_vmax
        )
        ax_abs.set_title(
            'Absolute Error (MAE & RMSE)',
            fontsize=13, fontweight='bold'
        )
        ax_abs.set_ylabel('Variables', fontsize=11)

        # 2c. Percentage error – MAPE (0–100%)
        sns.heatmap(
            error_pct,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            cbar_kws={
                'label': 'MAPE 0–100% (lebih kecil lebih baik)'
            },
            ax=ax_pct,
            linewidths=1,
            linecolor='white',
            vmin=0.0,
            vmax=100.0
        )
        ax_pct.set_title(
            'Percentage Error (MAPE)',
            fontsize=13, fontweight='bold'
        )
        ax_pct.set_ylabel('Variables', fontsize=11)

        fig.suptitle(
            'Model Evaluation Summary: R², MAE/RMSE, dan MAPE',
            fontsize=16, fontweight='bold', y=1.02
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'metrics_heatmaps_full.png'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
        plt.close()

        print(f"  ✓ Matriks komparasi disimpan di: {save_dir}/")
        print("    - model_comparison_matrix.png (tabel metrik)")
        print("    - metrics_heatmaps_full.png (R² + error dalam 1 gambar)")


# ============================= PIPELINE TRAINING =============================


def process_csv_file(csv_path, trainer, optimize=True):
    """Proses satu file CSV"""
    file_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"\n{'='*70}")
    print(f"MEMPROSES: {file_name.upper()}")
    print(f"{'='*70}")

    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset dimuat: {df.shape[0]} records")
    except Exception as error:
        print(f"Error: {error}")
        return None

    # Validasi kolom
    required_cols = ['timestamp', 'suhu', 'kelembapan']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Kolom tidak lengkap. Dibutuhkan: {required_cols}")
        return None

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    print("Resampling ke frekuensi per menit...")
    df = df.resample('T').interpolate(method='time')
    print(f"Setelah resampling: {df.shape[0]} records")

    df['suhu'] = df['suhu'].interpolate(method='time')
    df['kelembapan'] = df['kelembapan'].interpolate(method='time')

    # Sliding window
    df = trainer.apply_sliding_window(df)

    # Outlier handling
    print("\nMembersihkan outlier...")
    df['suhu'] = trainer.detect_and_remove_outliers(df['suhu'], 'suhu')
    df['kelembapan'] = trainer.detect_and_remove_outliers(df['kelembapan'], 'kelembapan')

    # Cek stasioneritas
    print("\nPengecekan stasioneritas:")
    trainer.check_stationarity(df['suhu'], 'suhu')
    trainer.check_stationarity(df['kelembapan'], 'kelembapan')

    # Folder output
    data_parent = os.path.dirname(os.path.dirname(csv_path))
    plots_dir = os.path.join(data_parent, "plots", file_name)
    os.makedirs(plots_dir, exist_ok=True)

    models = {}
    all_metrics = {}
    models_info = {}

    # Training untuk setiap variabel
    for variable in ['suhu', 'kelembapan']:
        print(f"\n{'='*50}")
        print(f"TRAINING: {variable.upper()}")
        print(f"{'='*50}")

        if optimize:
            order, seasonal = trainer.optimize_sarima_params(df, variable)
        else:
            order = (1, 1, 1)
            seasonal = (0, 0, 0, 0)
            print(f"Parameter default: ARIMA{order}")

        if order is None:
            print(f"✗ Optimasi gagal, skip {variable}...")
            continue

        model = trainer.train_model(df, variable, order, seasonal)

        if model is None:
            print(f"✗ Training {variable} gagal, skip...")
            continue

        # Hitung metrik (pakai fitted values)
        fitted_vals = model.fittedvalues
        actual_vals = df[variable].values[-len(fitted_vals):]
        metrics = trainer.calculate_metrics(actual_vals, fitted_vals.values, variable)

        all_metrics[variable] = metrics
        models[variable] = model
        models_info[variable] = {
            'model': model,
            'order': order,
            'seasonal': seasonal,
            'metrics': metrics,
            'rmse': metrics['rmse']
        }

        trainer.display_metrics(metrics, variable, "Training")

        # Validasi dengan 180 data terakhir (forecast 180 menit)
        print("\nValidasi dengan 180 data terakhir:")
        validation_metrics = trainer.calculate_metrics(
            df[variable].values[-180:],
            model.forecast(steps=180).values,
            variable
        )
        trainer.display_metrics(validation_metrics, variable, "Validation (180 menit)")

    # Plot Model Comparison Matrix (jika keduanya berhasil)
    if len(models_info) == 2:
        trainer.plot_model_comparison_matrix(models_info, df, plots_dir)

    # Simpan model
    model_data = {
        'models': models,
        'metrics': all_metrics,
        'last_training': datetime.now(),
        'window_days': trainer.window_days,
        'frequency': 'per_minute',
        'source_file': file_name,
        'plots_directory': plots_dir
    }

    models_dir = os.path.join(data_parent, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_file = os.path.join(models_dir, f"{file_name}_model.pkl")

    try:
        with open(model_file, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"\n✓ Model disimpan: {model_file}")
        print(f"✓ Matriks komparasi tersimpan di: {plots_dir}")
    except Exception as error:
        print(f"✗ Error menyimpan model: {error}")
        return None

    return model_data


def find_csv_files(data_dir):
    return glob.glob(os.path.join(data_dir, "*.csv"))


def print_summary(results):
    """Ringkasan akhir"""
    print("\n" + "="*70)
    print("RINGKASAN AKHIR TRAINING")
    print("="*70)

    for file_name, result in results.items():
        print(f"\n{'='*70}")
        print(f"FILE: {file_name.upper()}")
        print(f"{'='*70}")

        if 'metrics' in result:
            for variable, metrics in result['metrics'].items():
                print(f"\n{variable.upper()}:")
                print(
                    f"  R²: {metrics['r_squared']:.4f} | "
                    f"MAE: {metrics['mae']:.4f} ({metrics['mae_score']}) | "
                    f"RMSE: {metrics['rmse']:.4f} ({metrics['rmse_score']})"
                )
                print(
                    f"  MAPE: {metrics['mape']:.2f}% | "
                    f"Performance (R²): {metrics['performance']}"
                )


def main():
    print("="*70)
    print("SISTEM TRAINING SARIMA v3.3 (Metrics Matrix + Combined Heatmaps)")
    print("="*70)
    print(f"Dimulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("⚠ Tidak ada plot time-series, fokus ke evaluasi metrik.\n")

    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "data")
    csv_files = find_csv_files(data_dir)

    if not csv_files:
        print(f"✗ Tidak ada file CSV di {data_dir}")
        return

    print(f"\nDitemukan {len(csv_files)} file CSV:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(f)}")

    trainer = SARIMATrainer(window_days=7, outlier_threshold=3.0)

    results = {}
    success = 0

    for csv_file in csv_files:
        try:
            result = process_csv_file(csv_file, trainer, optimize=True)
            if result:
                results[os.path.splitext(os.path.basename(csv_file))[0]] = result
                success += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print("\n" + "="*70)
    print("TRAINING SELESAI")
    print("="*70)
    print(
        f"Total: {len(csv_files)} file | "
        f"Berhasil: {success} | Gagal: {len(csv_files) - success}"
    )

    if results:
        print_summary(results)

    print(f"\nSelesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
