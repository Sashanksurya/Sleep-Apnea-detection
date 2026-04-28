import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import os

FEATURE_ORDER = [
    "ecg_mean","ecg_std","ecg_min","ecg_max","ecg_range",
    "ecg_skew","ecg_kurtosis","ecg_energy","ecg_zcr",
    "rr_mean","rr_std","rr_min","rr_max","rmssd","heart_rate",
    "resp_mean","resp_std","resp_min","resp_max","resp_range",
    "resp_skew","resp_kurtosis","resp_variance",
    "resp_below90","resp_below94","resp_low_ratio","resp_drop_count",
]

def extract_segment_features(segment, fs=250):
    ecg, resp = segment[:, 0], segment[:, 1]
    f = OrderedDict()
    # ECG Stats
    f["ecg_mean"], f["ecg_std"] = np.mean(ecg), np.std(ecg)
    f["ecg_min"], f["ecg_max"] = np.min(ecg), np.max(ecg)
    f["ecg_range"] = f["ecg_max"] - f["ecg_min"]
    f["ecg_skew"], f["ecg_kurtosis"] = skew(ecg), kurtosis(ecg)
    f["ecg_energy"] = np.mean(ecg**2)
    f["ecg_zcr"] = np.sum(np.diff(np.sign(ecg)) != 0) / len(ecg)
    # Heart Rate
    peaks, _ = find_peaks(ecg, distance=int(fs*0.5), height=0.3)
    if len(peaks) > 1:
        rr = np.diff(peaks) / fs * 1000
        f["rr_mean"], f["rr_std"] = np.mean(rr), np.std(rr)
        f["rr_min"], f["rr_max"] = np.min(rr), np.max(rr)
        f["rmssd"] = np.sqrt(np.mean(np.diff(rr)**2))
        f["heart_rate"] = 60000 / np.mean(rr)
    else:
        for k in ["rr_mean","rr_std","rr_min","rr_max","rmssd","heart_rate"]: f[k] = 0.0
    # Resp Stats
    f["resp_mean"], f["resp_std"] = np.mean(resp), np.std(resp)
    f["resp_min"], f["resp_max"] = np.min(resp), np.max(resp)
    f["resp_range"] = f["resp_max"] - f["resp_min"]
    f["resp_skew"], f["resp_kurtosis"] = skew(resp), kurtosis(resp)
    f["resp_variance"] = np.var(resp)
    f["resp_below90"], f["resp_below94"] = np.mean(resp < 0.90), np.mean(resp < 0.94)
    f["resp_low_ratio"] = np.mean(resp < 0.95)
    f["resp_drop_count"] = float(len(find_peaks(-resp, prominence=0.02)[0]))
    return f

if __name__ == "__main__":
    os.makedirs("data/features", exist_ok=True)
    for split in ["train", "val", "test"]:
        X, y = np.load(f"data/processed/X_{split}.npy"), np.load(f"data/processed/y_{split}.npy")
        rows = [extract_segment_features(seg) for seg in X]
        df = pd.DataFrame(rows, columns=FEATURE_ORDER).fillna(0.0)
        df.to_csv(f"data/features/X_{split}_feat.csv", index=False)
        np.save(f"data/features/y_{split}.npy", y)
    print("✅ Phase 3 Complete: Canonical features extracted.")