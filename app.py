from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import timedelta  # boleh tetap, meski kita pakai tahun langsung
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Untuk data tahunan, window jangan terlalu besar
WINDOW_SIZE = 3  # gunakan 3 titik terakhir sebagai fitur


def make_supervised(series, window=WINDOW_SIZE):
    """
    Mengubah data time series menjadi supervised learning.
    Input: [v1, v2, ..., vn]
    Output: X = [v1..v_window], [v2..v_window+1], ...
            y = nilai setelah window.
    """
    X, y = [], []
    values = series.values
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)


def train_model(df, model_type):
    """
    Melatih model sesuai pilihan user.
    df sudah diurutkan berdasarkan tanggal dan punya kolom 'volume'.
    """
    volume_series = df["volume"]

    # siapkan data supervised
    X, y = make_supervised(volume_series, window=WINDOW_SIZE)
    if len(X) == 0:
        raise ValueError("Data terlalu sedikit untuk dilatih (minimal window+1 baris).")

    # ==========================
    # DAFTAR MODEL YANG DIDUKUNG
    # ==========================
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001)
    }

    if model_type not in models:
        raise ValueError(f"Jenis model '{model_type}' tidak dikenal.")

    model = models[model_type]
    model.fit(X, y)
    return model


def predict_future(df, model, years_ahead, start_year=None):
    """
    Prediksi 'years_ahead' ke depan dari data df.

    - Dataset tahunan: 1 baris = 1 tahun
    - 1 langkah prediksi = 1 tahun
    - Jika start_year diisi:
        * window awal diambil dari WINDOW_SIZE tahun sebelum/tahun itu
        * prediksi dimulai dari tahun start_year + 1
    - Jika start_year kosong:
        * mulai dari tahun terakhir di dataset + 1
    """
    df = df.sort_values("tanggal").copy()

    # pastikan ada kolom 'tahun'
    if "tahun" not in df.columns:
        df["tahun"] = df["tanggal"].dt.year

    years = df["tahun"].values
    volumes = df["volume"].values

    # --- tentukan titik awal (last_year & last_values) ---
    if start_year is not None:
        idx = np.where(years == start_year)[0]
        if len(idx) == 0:
            raise ValueError(f"Tahun awal {start_year} tidak ditemukan di dataset.")

        pos = idx[-1]  # posisi index tahun awal
        if pos < WINDOW_SIZE - 1:
            raise ValueError(
                f"Data sebelum tahun {start_year} kurang dari {WINDOW_SIZE} tahun, "
                f"tidak bisa dijadikan window."
            )

        last_year = years[pos]  # misal 2020
        # window: beberapa tahun terakhir sampai tahun start_year
        last_values = volumes[pos - WINDOW_SIZE + 1: pos + 1]
    else:
        # default: mulai dari tahun terakhir di dataset
        last_year = years[-1]      # misal 2023
        last_values = volumes[-WINDOW_SIZE:]

    preds = []
    current_window = last_values.copy()

    # --- langkah prediksi per tahun ---
    for _ in range(years_ahead):
        x_input = current_window.reshape(1, -1)
        pred = model.predict(x_input)[0]

        future_year = last_year + 1  # 1 langkah = 1 tahun ke depan
        preds.append(
            {
                "tanggal": f"{future_year}-01-01",  # tampilkan sebagai 1 Jan tiap tahun
                "prediksi_volume": round(float(pred), 2),
            }
        )

        # update untuk langkah berikutnya
        last_year = future_year
        current_window = np.append(current_window[1:], pred)

    return preds


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # ambil input form
    model_type = request.form.get("model_type")
    years_ahead = int(request.form.get("years_ahead", 1))

    start_year_raw = request.form.get("start_year", "").strip()
    start_year = int(start_year_raw) if start_year_raw else None

    file = request.files.get("dataset")
    if file is None or file.filename == "":
        return "Dataset belum diupload", 400

    # Baca CSV
    df = pd.read_csv(file)

    # Standarisasi nama kolom ke huruf kecil & hilangkan spasi
    df.columns = [c.strip().lower() for c in df.columns]

    # Jika dataset punya 'tahun' dan 'volume_ton', konversi ke 'tanggal' & 'volume'
    if "tahun" in df.columns and "volume_ton" in df.columns and "tanggal" not in df.columns:
        df["tanggal"] = pd.to_datetime(df["tahun"].astype(str) + "-01-01")
        df["volume"] = df["volume_ton"]

    # Pastikan kolom minimum yang dibutuhkan ada
    required_cols = {"tanggal", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return (
            "Kolom CSV harus mengandung minimal: tanggal, volume "
            "(atau tahun, volume_ton yang akan dikonversi)",
            400,
        )

    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal")

    # Nama TPA (opsional)
    if "tpa" in df.columns:
        tpa_name = str(df["tpa"].iloc[0])
    else:
        tpa_name = "TPA"

    try:
        model = train_model(df, model_type)
        preds = predict_future(df, model, years_ahead, start_year=start_year)
    except Exception as e:
        return f"Terjadi error saat training/prediksi: {str(e)}", 500

    history_tail = df.tail(10)

    return render_template(
        "result.html",
        tpa_name=tpa_name,
        model_type=model_type,
        years_ahead=years_ahead,
        start_year=start_year,
        preds=preds,
        history=history_tail.to_dict(orient="records"),
    )


if __name__ == "__main__":
    app.run(debug=True)
