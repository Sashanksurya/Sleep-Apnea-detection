from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib, os, traceback, wfdb
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from collections import OrderedDict

# ── Resolve project root robustly ──
_HERE = os.path.dirname(os.path.abspath(__file__))
# Walk up until we find a folder that contains both "data" and "model"
ROOT = _HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "data")) and os.path.isdir(os.path.join(ROOT, "model")):
        break
    ROOT = os.path.dirname(ROOT)

RAW_DIR  = os.path.join(ROOT, "data", "raw")
ML_PATH  = os.path.join(ROOT, "model", "rf_model.pkl")
DL_PATH  = os.path.join(ROOT, "model", "cnn_lstm_final.keras")
TMPL_DIR = os.path.join(_HERE, "templates")

print(f"[BOOT] ROOT    = {ROOT}")
print(f"[BOOT] RAW_DIR = {RAW_DIR}  exists={os.path.isdir(RAW_DIR)}")
print(f"[BOOT] ML_PATH = {ML_PATH}  exists={os.path.exists(ML_PATH)}")

app = Flask(__name__, template_folder=TMPL_DIR)

# ---------------- MODEL LOADING ----------------
ml_model = joblib.load(ML_PATH) if os.path.exists(ML_PATH) else None
dl_model = None
try:
    import tensorflow as tf
    dl_model = tf.keras.models.load_model(DL_PATH) if os.path.exists(DL_PATH) else None
except Exception as e:
    print(f"[WARN] Could not load DL model: {e}")

# ---------------- CONSTANTS ----------------
WINDOW = 15000
FS_DEF = 250.0

FEATURE_ORDER = [
    "ecg_mean","ecg_std","ecg_min","ecg_max","ecg_range",
    "ecg_skew","ecg_kurtosis","ecg_energy","ecg_zcr",
    "rr_mean","rr_std","rr_min","rr_max","rmssd","heart_rate",
    "resp_mean","resp_std","resp_min","resp_max","resp_range",
    "resp_skew","resp_kurtosis","resp_variance",
    "resp_below90","resp_below94","resp_low_ratio","resp_drop_count",
]

# ---------------- SIGNAL UTILITIES ----------------
def _fill_nan(arr):
    arr = np.array(arr, dtype=np.float64)
    mask = ~np.isfinite(arr)
    if mask.all(): return np.zeros_like(arr)
    if mask.any(): arr[mask] = np.nanmean(arr[~mask])
    return arr

def _normalize(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn) if mx > mn else np.zeros_like(v)

def _safe_bandpass(x, fs, lo=0.5, hi=40.0):
    try:
        nyq = 0.5 * fs
        b, a = butter(4, [max(lo/nyq, 1e-4), min(hi/nyq, 0.9999)], btype="band")
        if len(x) <= 3 * max(len(a), len(b)): return x
        return filtfilt(b, a, x)
    except Exception: return x

def _safe_lowpass(x, fs, cutoff=1.0):
    try:
        nyq = 0.5 * fs
        b, a = butter(2, min(cutoff/nyq, 0.9999), btype="low")
        if len(x) <= 3 * max(len(a), len(b)): return x
        return filtfilt(b, a, x)
    except Exception: return x

def _find_channel(sig_names, keywords):
    upper = [s.upper().strip() for s in sig_names]
    for kw in keywords:
        for i, name in enumerate(upper):
            if kw in name: return i
    return None

def _pick_channels(sig_names, signals):
    n_ch = signals.shape[1]
    ecg_idx = _find_channel(sig_names, ["ECG","EKG","II ","LEAD"])
    if ecg_idx is None: ecg_idx = 0
    resp_idx = _find_channel(sig_names, ["SO2","SPO2","SAO2","O2SAT"])
    has_spo2 = resp_idx is not None
    if not has_spo2:
        resp_idx = _find_channel(sig_names, ["RESP","NASAL","CHEST","ABDO","SUM","FLOW"])
    if resp_idx is None or resp_idx == ecg_idx:
        candidates = [i for i in range(n_ch) if i != ecg_idx]
        resp_idx = candidates[0] if candidates else ecg_idx
    return (np.asarray(signals[:, ecg_idx], dtype=np.float64),
            np.asarray(signals[:, resp_idx], dtype=np.float64),
            has_spo2)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(ecg, resp, fs=FS_DEF):
    from scipy.stats import skew, kurtosis as kurt
    f = OrderedDict()
    f["ecg_mean"]     = float(np.mean(ecg))
    f["ecg_std"]      = float(np.std(ecg))
    f["ecg_min"]      = float(np.min(ecg))
    f["ecg_max"]      = float(np.max(ecg))
    f["ecg_range"]    = f["ecg_max"] - f["ecg_min"]
    f["ecg_skew"]     = float(skew(ecg))
    f["ecg_kurtosis"] = float(kurt(ecg))
    f["ecg_energy"]   = float(np.mean(ecg**2))
    f["ecg_zcr"]      = float(np.sum(np.diff(np.sign(ecg)) != 0) / max(len(ecg)-1, 1))
    peaks, _ = find_peaks(ecg, distance=int(fs*0.5), height=0.3)
    if len(peaks) > 1:
        rr = np.diff(peaks) / fs * 1000
        f["rr_mean"]    = float(np.mean(rr))
        f["rr_std"]     = float(np.std(rr))
        f["rr_min"]     = float(np.min(rr))
        f["rr_max"]     = float(np.max(rr))
        f["rmssd"]      = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else 0.0
        f["heart_rate"] = float(60000.0 / np.mean(rr))
    else:
        for k in ["rr_mean","rr_std","rr_min","rr_max","rmssd","heart_rate"]: f[k] = 0.0
    f["resp_mean"]      = float(np.mean(resp))
    f["resp_std"]       = float(np.std(resp))
    f["resp_min"]       = float(np.min(resp))
    f["resp_max"]       = float(np.max(resp))
    f["resp_range"]     = f["resp_max"] - f["resp_min"]
    f["resp_skew"]      = float(skew(resp))
    f["resp_kurtosis"]  = float(kurt(resp))
    f["resp_variance"]  = float(np.var(resp))
    f["resp_below90"]   = float(np.mean(resp < 0.90))
    f["resp_below94"]   = float(np.mean(resp < 0.94))
    f["resp_low_ratio"] = float(np.mean(resp < 0.95))
    f["resp_drop_count"]= float(len(find_peaks(-resp, prominence=0.02)[0]))
    return f

# ---------------- CORE PREDICTION ----------------
def run_prediction(ecg_raw, resp_raw, fs=FS_DEF, has_spo2=True):
    W = WINDOW
    ecg_raw  = _fill_nan(np.asarray(ecg_raw,  dtype=np.float64))
    resp_raw = _fill_nan(np.asarray(resp_raw, dtype=np.float64))
    length   = min(len(ecg_raw), len(resp_raw))
    ecg_raw  = ecg_raw[:length]
    resp_raw = resp_raw[:length]
    resp_unit = resp_raw / 100.0 if np.nanmax(np.abs(resp_raw)) > 5.0 else resp_raw.copy()
    ecg_filt  = _safe_bandpass(ecg_raw,  fs)
    resp_filt = _safe_lowpass(resp_unit, fs, cutoff=1.0)
    ecg_win  = np.zeros(W, dtype=np.float64)
    resp_win = np.zeros(W, dtype=np.float64)
    fill = min(W, length)
    ecg_win[:fill]  = _normalize(ecg_filt[:fill])
    resp_win[:fill] = _normalize(resp_filt[:fill])
    feats = extract_features(ecg_win, resp_win, fs)
    fvec  = np.nan_to_num(
        np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0
    ).reshape(1, -1)
    ml_prob = 0.5
    if ml_model:
        try: ml_prob = float(ml_model.predict_proba(fvec)[0][1])
        except Exception as e: print(f"[ML] {e}")
    dl_prob = 0.5
    if dl_model:
        try:
            x_dl = np.column_stack([ecg_win, resp_win]).reshape(1, W, 2).astype(np.float32)
            dl_prob = float(dl_model.predict(x_dl, verbose=0)[0][0])
        except Exception as e: print(f"[DL] {e}")
    final_prob   = (ml_prob + dl_prob) / 2.0
    spo2_display = resp_unit * 100.0
    return {
        "prediction":  int(final_prob > 0.5),
        "probability": round(final_prob * 100, 2),
        "risk_level":  "HIGH" if final_prob > 0.7 else "MEDIUM" if final_prob > 0.3 else "LOW",
        "heart_rate":  round(feats["heart_rate"], 1),
        "spo2_mean":   round(float(np.mean(spo2_display)), 1),
        "spo2_min":    round(float(np.min(spo2_display)),  1),
        "rf_prob":     round(ml_prob  * 100, 2),
        "dl_prob":     round(dl_prob  * 100, 2),
    }

# ================================================================
# ROUTES
# ================================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/debug")
def debug():
    """Shows exactly where app.py is looking for files."""
    hea_files = []
    if os.path.isdir(RAW_DIR):
        hea_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".hea")]
    return jsonify({
        "ROOT":     ROOT,
        "RAW_DIR":  RAW_DIR,
        "exists":   os.path.isdir(RAW_DIR),
        "hea_files": sorted(hea_files),
        "ML_PATH":  ML_PATH,
        "ml_exists": os.path.exists(ML_PATH),
    })

@app.route("/list_records")
def list_records():
    try:
        if not os.path.isdir(RAW_DIR):
            return jsonify({"error": f"RAW_DIR not found: {RAW_DIR}"}), 500
        recs = sorted(f.replace(".hea", "") for f in os.listdir(RAW_DIR) if f.endswith(".hea"))
        return jsonify(recs)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route("/predict_raw/<filename>")
def predict_raw(filename):
    rec_name = filename.split(".")[0]
    rec_path = os.path.join(RAW_DIR, rec_name)
    try:
        record  = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs      = float(record.fs)
        names   = list(record.sig_name)
        print(f"[RECORD] {rec_name}  fs={fs}  shape={signals.shape}  channels={names}")
        if signals is None or signals.ndim != 2 or signals.shape[0] == 0:
            return jsonify({"error": "Record has no usable signal data."}), 400
        if signals.shape[1] == 1:
            ecg_arr  = np.asarray(signals[:, 0], dtype=np.float64)
            resp_arr = np.zeros_like(ecg_arr)
            has_spo2 = False
        else:
            ecg_arr, resp_arr, has_spo2 = _pick_channels(names, signals)
        result = run_prediction(ecg_arr, resp_arr, fs=fs, has_spo2=has_spo2)
        print(f"[RESULT] {result}")
        return jsonify(result)
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] /predict_raw/{filename}\n{tb}")
        return jsonify({"error": tb}), 500

@app.route("/predict_hea", methods=["POST"])
def predict_hea():
    """
    Upload a .hea + .dat (+ optional .st/.apn) file set.
    The user uploads just the .hea; we save it and its companion .dat
    to a temp folder, then run wfdb on it.
    """
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    try:
        saved = {}
        for key, f in request.files.items():
            fname = f.filename
            dest  = os.path.join(tmp_dir, os.path.basename(fname))
            f.save(dest)
            saved[fname] = dest
            print(f"[UPLOAD] {fname} → {dest}")

        # Find the .hea file to get the record name
        hea_files = [k for k in saved if k.endswith(".hea")]
        if not hea_files:
            return jsonify({"error": "No .hea file found in upload."}), 400

        rec_name = os.path.splitext(os.path.basename(hea_files[0]))[0]
        rec_path = os.path.join(tmp_dir, rec_name)

        # Check .dat exists
        dat_path = rec_path + ".dat"
        if not os.path.exists(dat_path):
            return jsonify({"error": f".dat file missing. Please upload both {rec_name}.hea and {rec_name}.dat"}), 400

        record  = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs      = float(record.fs)
        names   = list(record.sig_name)

        # equalise lengths
        ecg_arr, resp_arr, has_spo2 = _pick_channels(names, signals) if signals.shape[1] > 1 else (
            np.asarray(signals[:, 0], dtype=np.float64), np.zeros(signals.shape[0]), False)

        result = run_prediction(ecg_arr, resp_arr, fs=fs, has_spo2=has_spo2)
        result["record_name"] = rec_name
        return jsonify(result)

    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] /predict_hea\n{tb}")
        return jsonify({"error": tb}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route("/predict", methods=["POST"])
def predict_csv():
    try:
        f = request.files.get("signal_file")
        if not f:
            return jsonify({"error": "No file uploaded."}), 400
        import io
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8", errors="replace")))
        df.columns = [c.strip().lower() for c in df.columns]
        ecg_col  = next((c for c in df.columns if "ecg" in c), None)
        resp_col = next((c for c in df.columns if any(k in c for k in ["spo2","resp","spo","o2"])), None)
        if ecg_col is None:
            return jsonify({"error": f"No ECG column found. Columns: {list(df.columns)}"}), 400
        if resp_col is None:
            resp_col = df.columns[1] if len(df.columns) > 1 else ecg_col
        result = run_prediction(
            df[ecg_col].to_numpy(dtype=np.float64),
            df[resp_col].to_numpy(dtype=np.float64)
        )
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        f = request.files.get("ecg_image")
        if not f:
            return jsonify({"error": "No image uploaded."}), 400
        from PIL import Image
        import io as _io
        img      = Image.open(_io.BytesIO(f.read())).convert("L")
        img      = img.resize((WINDOW, 200))
        arr      = np.array(img, dtype=np.float64)
        trace_row= int(np.argmin(arr.mean(axis=1)))
        ecg_raw  = (255.0 - arr[trace_row]) / 255.0
        resp_raw = np.clip(np.full(WINDOW, 0.96) + np.random.normal(0, 0.005, WINDOW), 0.75, 1.0)
        result = run_prediction(ecg_raw, resp_raw)
        result["source"] = "image_heuristic"
        return jsonify(result)
    except ImportError:
        return jsonify({"error": "Run: pip install Pillow"}), 500
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)