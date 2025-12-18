# utils.py
import pandas as pd
import yfinance as yf
from pathlib import Path
import joblib
import os
from config import UPLOAD_DIR, SKLEARN_DIR, TF_DIR, REPORTS_DIR
from datetime import datetime

REQUIRED_OHLC = {"date","open","high","low","close","volume"}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # normalize to lowercase and underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    # aliases
    rename = {}
    for c in df.columns:
        if c in ("adj_close","adjclose","adj_close"):
            rename[c] = "adj_close"
        if c in ("timestamp","datetime"):
            rename[c] = "date"
    if rename:
        df = df.rename(columns=rename)
    return df

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # try mapping common variants
    cols = set(df.columns)
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if "open" in lc and "open" not in cols:
            mapping[c] = "open"
        if "high" in lc and "high" not in cols:
            mapping[c] = "high"
        if "low" in lc and "low" not in cols:
            mapping[c] = "low"
        if "close" in lc and "close" not in cols:
            mapping[c] = "close"
        if "volume" in lc and "volume" not in cols:
            mapping[c] = "volume"
        if ("date" in lc or "time" in lc) and "date" not in cols:
            mapping[c] = "date"
    if mapping:
        df = df.rename(columns=mapping)
    return df

def validate_ohlc(df: pd.DataFrame):
    present = set(df.columns)
    missing = list(REQUIRED_OHLC - present)
    return missing

def load_yfinance(ticker: str, period: str = "5y", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None, f"yfinance returned empty data for {ticker}"
        df = df.reset_index()
        df = _normalize_columns(df)
        df = _ensure_ohlc(df)
        # coerce date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        missing = validate_ohlc(df)
        if missing:
            return None, f"Missing required columns after normalization: {missing}"
        return df, None
    except Exception as e:
        return None, f"yfinance load error: {e}"

def load_csv(uploaded_path_or_buffer):
    try:
        df = pd.read_csv(uploaded_path_or_buffer)
        df = _normalize_columns(df)
        df = _ensure_ohlc(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        missing = validate_ohlc(df)
        if missing:
            return None, f"CSV missing required columns: {missing}"
        return df, None
    except Exception as e:
        return None, f"CSV load error: {e}"

def load_stock_data(source: str, ticker: str, period: str = "5y", interval: str = "1d", uploaded_file=None):
    """
    source: "yfinance" or "upload_csv"
    If upload_csv, uploaded_file is file-like or path
    Returns (df, error_msg)
    """
    if source == "yfinance":
        return load_yfinance(ticker, period, interval)
    elif source == "upload_csv":
        if uploaded_file is None:
            return None, "No CSV uploaded"
        return load_csv(uploaded_file)
    else:
        return None, "Invalid data source"

# model persistence
def save_sklearn_model(model, name: str):
    SKLEARN_DIR.mkdir(parents=True, exist_ok=True)
    path = SKLEARN_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path

def load_sklearn_model(name: str):
    path = SKLEARN_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return joblib.load(path)

def save_tf_model(model, name: str):
    TF_DIR.mkdir(parents=True, exist_ok=True)
    path = TF_DIR / name
    # tensorflow will create directory
    try:
        model.save(str(path), save_format='tf')
    except Exception:
        # fallback to HDF5
        path = TF_DIR / f"{name}.h5"
        model.save(str(path))
    return path

def load_tf_model(name: str):
    path_dir = TF_DIR / name
    if path_dir.exists():
        import tensorflow as tf
        return tf.keras.models.load_model(str(path_dir))
    # fallback h5
    path_h5 = TF_DIR / f"{name}.h5"
    if path_h5.exists():
        import tensorflow as tf
        return tf.keras.models.load_model(str(path_h5))
    raise FileNotFoundError("No TF model found")

# simple PDF report writer
def create_report_pdf(text: str, filename: str):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / filename
        c = canvas.Canvas(str(out), pagesize=letter)
        w, h = letter
        x, y = 50, h - 50
        for line in text.splitlines():
            c.drawString(x, y, line[:200])
            y -= 14
            if y < 50:
                c.showPage()
                y = h - 50
        c.save()
        return out
    except Exception as e:
        return None
