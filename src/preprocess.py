import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

os.makedirs("data/processed", exist_ok=True)

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, min(highcut/nyq, 0.99)], btype="band")
    return filtfilt(b, a, signal)

def lowpass_filter(signal, cutoff=1.0, fs=250, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff/nyq, 0.99), btype="low")
    return filtfilt(b, a, signal)

def normalize(signal):
    mn, mx = np.nanmin(signal), np.nanmax(signal)
    return (signal - mn) / (mx - mn) if mx != mn else np.zeros_like(signal)

def scale_resp_channel(signal):
    # MIT-BIH % scaling logic
    return signal / 100.0 if np.nanmax(signal) > 2.0 else signal.copy()

def preprocess_all(X, fs=250):
    X_clean = np.zeros_like(X)
    for i in range(X.shape[0]):
        ecg, resp = X[i, :, 0], scale_resp_channel(X[i, :, 1])
        # Clean NaN
        ecg[np.isnan(ecg)] = np.nanmean(ecg) if not np.isnan(ecg).all() else 0
        resp[np.isnan(resp)] = np.nanmean(resp) if not np.isnan(resp).all() else 0
        # Filter & Normalize
        X_clean[i, :, 0] = normalize(bandpass_filter(ecg, fs=fs))
        X_clean[i, :, 1] = normalize(lowpass_filter(resp, fs=fs))
    return X_clean

if __name__ == "__main__":
    X, y = np.load("data/X_raw.npy"), np.load("data/y_raw.npy")
    X_clean = preprocess_all(X)
    # Balanced Resampling
    X0, X1 = X_clean[y==0], X_clean[y==1]
    y0, y1 = y[y==0], y[y==1]
    target = max(len(X0), len(X1))
    X0, y0 = resample(X0, y0, n_samples=target, random_state=42)
    X1, y1 = resample(X1, y1, n_samples=target, random_state=42)
    X_bal, y_bal = np.concatenate([X0, X1]), np.concatenate([y0, y1])
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.125, stratify=y_tr)
    # Save
    for name, data in [("X_train", X_tr), ("X_val", X_val), ("X_test", X_te), 
                       ("y_train", y_tr), ("y_val", y_val), ("y_test", y_te)]:
        np.save(f"data/processed/{name}.npy", data)
    print("✅ Phase 2 Complete: Signals filtered and balanced.")