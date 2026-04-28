# ============================================================
# PHASE 1 — DATASET LOADING
# ============================================================
# WHAT THIS FILE DOES:
#   Loads raw signals (ECG + SpO2/Respiration) from the MIT-BIH
#   Polysomnographic Database, reads apnea annotations,
#   cuts signals into 60-second windows, and saves them as
#   numpy arrays (X_raw.npy, y_raw.npy) for the next phase.
#
# DATASET: MIT-BIH Polysomnographic Database (PhysioNet)
#   https://physionet.org/content/slpdb/1.0.0/
#   18 overnight sleep recordings of real patients
#   Each record has: ECG, Blood Pressure, EEG, Respiration, SpO2
#   Annotations mark each minute as Apnea (A) or Normal (N)
#
# DOWNLOAD:
#   pip install wfdb
#   python -c "import wfdb; wfdb.dl_database('slpdb', 'data/raw')"
#   OR manually from https://physionet.org/content/slpdb/1.0.0/
#
# RUN: python src/load_data.py
# ============================================================

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR   = "data/raw"
OUTPUT_DIR = "outputs/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data",     exist_ok=True)

# All 18 records in the MIT-BIH Polysomnographic database
RECORD_NAMES = [
    "slp01a","slp01b","slp02a","slp02b","slp03","slp04",
    "slp14", "slp16", "slp32", "slp37", "slp41", "slp45",
    "slp48", "slp59", "slp60", "slp61", "slp66", "slp67x"
]

# ── Helper: find which column index contains a named signal ──
def find_channel(sig_names, keywords):
    upper = [s.upper().strip() for s in sig_names]
    for kw in keywords:
        for i, name in enumerate(upper):
            if kw in name:
                return i
    return None

# ── Load one record and extract ECG + SpO2/Resp ──
def load_record(record_name):
    path   = os.path.join(DATA_DIR, record_name)
    record = wfdb.rdrecord(path)
    print(f"  Channels : {record.sig_name}")
    print(f"  Fs       : {record.fs} Hz  |  Length: {record.sig_len} samples  "
          f"({record.sig_len / record.fs / 3600:.2f} h)")

    # ECG channel
    ecg_idx = find_channel(record.sig_name, ["ECG", "EKG", "II", "I"])
    if ecg_idx is None:
        ecg_idx = 0
        print(f"  [WARN] No ECG channel found; using column 0 ({record.sig_name[0]})")

    # SpO2 channel (preferred)
    spo2_idx  = find_channel(record.sig_name, ["SO2", "SPO2", "SAO2", "O2SAT", "O2"])
    has_spo2  = spo2_idx is not None

    if not has_spo2:
        # Fallback: respiration (airflow / chest / abdomen)
        spo2_idx = find_channel(record.sig_name, ["RESP", "NASAL", "CHEST", "ABDO", "SUM"])
        if spo2_idx is None:
            # Last resort: pick any remaining channel that is not ECG
            candidates = [i for i in range(record.n_sig) if i != ecg_idx]
            spo2_idx   = candidates[0] if candidates else ecg_idx
        print(f"  [INFO] No SpO2 found; using {record.sig_name[spo2_idx]} as respiratory proxy")

    ch2_name = record.sig_name[spo2_idx]
    print(f"  ECG  → col {ecg_idx} ({record.sig_name[ecg_idx]})")
    print(f"  Ch2  → col {spo2_idx} ({ch2_name}) {'[SpO₂]' if has_spo2 else '[Resp fallback]'}")

    ecg  = record.p_signal[:, ecg_idx].astype(np.float32)
    spo2 = record.p_signal[:, spo2_idx].astype(np.float32)

    # MIT-BIH SpO2 is stored as percentage (0–100); keep it that way.
    # Respiration proxies may be in mV — we leave them raw; Phase 2 normalises.
    return ecg, spo2, record.fs, has_spo2

# ── Load apnea annotations (.apn file = real, per-minute apnea labels) ──
def load_annotations(record_name):
    path = os.path.join(DATA_DIR, record_name)

    # .apn — actual apnea annotations: A = apnea, N = normal
    try:
        ann    = wfdb.rdann(path, "apn")
        labels = [1 if s.strip().upper() == "A" else 0 for s in ann.symbol]
        apnea  = sum(labels)
        total  = len(labels)
        print(f"  Labels (.apn): {total} windows | Apnea={apnea} ({apnea/total*100:.1f}%)")
        return np.array(ann.sample, dtype=np.int64), np.array(labels, dtype=np.int8)
    except Exception:
        pass

    # .st — sleep-stage annotations (secondary option)
    try:
        ann    = wfdb.rdann(path, "st")
        labels = [1 if s.strip().upper() in ["A", "H", "OA", "MA", "CA", "X"] else 0
                  for s in ann.symbol]
        apnea  = sum(labels)
        total  = len(labels)
        print(f"  Labels (.st): {total} windows | Apnea={apnea} ({apnea/total*100:.1f}%)")
        return np.array(ann.sample, dtype=np.int64), np.array(labels, dtype=np.int8)
    except Exception:
        pass

    print(f"  [WARN] No annotation file found for {record_name}; using signal heuristic")
    return None, None

# ── Heuristic label generation when no annotation file is present ──
def generate_labels_from_signal(ecg, resp, fs, window_sec=60, has_spo2=False):
    """
    When real annotations are unavailable, infer labels from signal characteristics.
    SpO2 channel: apnea if mean < 93% or significant drop.
    Resp channel:  apnea if amplitude is very low (flat breathing).
    """
    window    = int(fs * window_sec)
    n_windows = len(ecg) // window
    samples, labels = [], []

    # Statistics on the full recording to set adaptive thresholds
    resp_clean  = resp[~np.isnan(resp)]
    global_mean = np.nanmean(resp_clean)
    global_std  = np.nanstd(resp_clean)

    for i in range(n_windows):
        start     = i * window
        seg_resp  = resp[start:start + window]
        seg_resp  = seg_resp[~np.isnan(seg_resp)]
        if len(seg_resp) == 0:
            samples.append(start); labels.append(0); continue

        seg_mean  = np.nanmean(seg_resp)
        seg_std   = np.nanstd(seg_resp)

        if has_spo2:
            # SpO2 in percent: below 93% mean = concerning
            is_apnea = 1 if seg_mean < 93.0 else 0
        else:
            # Respiratory proxy: very flat signal = no breathing
            is_apnea = 1 if (seg_std < global_std * 0.25 or
                              abs(seg_mean) < abs(global_mean) * 0.1) else 0

        samples.append(start)
        labels.append(is_apnea)

    apnea = sum(labels)
    total = len(labels)
    print(f"  Labels (auto): {total} windows | Apnea={apnea} ({apnea/total*100:.1f}%)")
    return np.array(samples, dtype=np.int64), np.array(labels, dtype=np.int8)

# ── Cut signals into 60-second windows ──
def segment_signals(ecg, spo2, samples, labels, fs, window_sec=60):
    window = int(fs * window_sec)   # 250 Hz × 60 s = 15 000 samples
    segs, lbls, skipped = [], [], 0

    for start, label in zip(samples, labels):
        end = start + window
        if end > len(ecg):
            skipped += 1
            continue
        e = ecg[start:end].astype(np.float32)
        s = spo2[start:end].astype(np.float32)
        # Skip windows with too many NaN values (>5%)
        if np.isnan(e).mean() > 0.05 or np.isnan(s).mean() > 0.05:
            skipped += 1
            continue
        # Fill remaining NaN with linear interpolation (rare)
        if np.isnan(e).any():
            e = pd.Series(e).interpolate(limit_direction="both").values.astype(np.float32)
        if np.isnan(s).any():
            s = pd.Series(s).interpolate(limit_direction="both").values.astype(np.float32)
        segs.append(np.column_stack((e, s)))   # shape: (15000, 2)
        lbls.append(label)

    print(f"  Segments: {len(segs)} accepted | {skipped} skipped")
    return np.array(segs, dtype=np.float32), np.array(lbls, dtype=np.int8)

# ── Plot 120 seconds of ECG and SpO2/Resp ──
def plot_signals(ecg, spo2, fs, name, has_spo2, sec=120):
    n = int(fs * sec)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"Signal Preview — {name}", fontsize=13, fontweight="bold")

    axes[0].plot(t, ecg[:n],  color="#38bdf8", lw=0.7)
    axes[0].set_ylabel("ECG (mV)")
    axes[0].set_title("ECG — Heart Signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, spo2[:n], color="#f87171", lw=1.0)
    if has_spo2:
        axes[1].axhline(y=90, color="#fbbf24", ls="--", lw=1, alpha=0.7, label="90% threshold")
        axes[1].legend()
    axes[1].set_ylabel("SpO₂ (%)" if has_spo2 else "Resp (mV)")
    axes[1].set_title("SpO₂ Signal (%)" if has_spo2 else "Respiration Signal")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{name}_signals.png")
    plt.savefig(out, dpi=120)
    print(f"  Plot saved → {out}")
    plt.show()

# ── Process ALL records and save combined dataset ──
def process_all_records():
    import pandas as pd  # only needed here for NaN interpolation
    all_segs, all_lbls, loaded = [], [], 0

    for rec in RECORD_NAMES:
        if not os.path.exists(os.path.join(DATA_DIR, rec + ".hea")):
            print(f"\n[SKIP] {rec} — .hea not found")
            continue
        print(f"\n{'─'*40}")
        print(f"  Record: {rec}")
        try:
            ecg, spo2, fs, has_spo2 = load_record(rec)
            samples, labels         = load_annotations(rec)

            # Use real annotations; fall back to heuristic only if none
            if samples is None or sum(labels) == 0:
                samples, labels = generate_labels_from_signal(ecg, spo2, fs, has_spo2=has_spo2)

            segs, lbls = segment_signals(ecg, spo2, samples, labels, fs)
            if len(segs) > 0:
                all_segs.append(segs)
                all_lbls.append(lbls)
                loaded += 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    if loaded == 0:
        print("\n[ERROR] No records loaded. Check DATA_DIR and dataset files.")
        return None, None

    X = np.concatenate(all_segs)
    y = np.concatenate(all_lbls)

    print(f"\n{'='*50}")
    print(f"  Records loaded : {loaded}")
    print(f"  Total segments : {X.shape[0]}")
    print(f"  Array shape    : {X.shape}  ({X.nbytes / 1024 / 1024:.0f} MB)")
    print(f"  Apnea  (1)     : {np.sum(y == 1)}")
    print(f"  Normal (0)     : {np.sum(y == 0)}")
    print(f"{'='*50}")

    np.save("data/X_raw.npy", X)
    np.save("data/y_raw.npy", y)
    print("  Saved → data/X_raw.npy  and  data/y_raw.npy")
    return X, y

if __name__ == "__main__":
    import pandas as pd  # for NaN interpolation in segment_signals

    TEST = "slp01a"
    hea  = os.path.join(DATA_DIR, TEST + ".hea")
    if not os.path.exists(hea):
        print(f"[ERROR] {hea} not found.\n"
              f"Download the dataset:\n"
              f"  python -c \"import wfdb; wfdb.dl_database('slpdb', '{DATA_DIR}')\"\n"
              f"or visit https://physionet.org/content/slpdb/1.0.0/")
    else:
        print("\n[Phase 1] Testing single record first...\n")
        ecg, spo2, fs, has_spo2 = load_record(TEST)
        samples, labels         = load_annotations(TEST)
        if samples is None or sum(labels) == 0:
            samples, labels = generate_labels_from_signal(ecg, spo2, fs, has_spo2=has_spo2)
        segs, lbls = segment_signals(ecg, spo2, samples, labels, fs)
        plot_signals(ecg, spo2, fs, TEST, has_spo2)
        print(f"\n  Test result: {len(lbls)} segments | Apnea: {sum(lbls)}")
        print("\n[OK] Processing all records...\n")
        X, y = process_all_records()
        if X is not None:
            print("\n✅ Phase 1 DONE → run: python src/preprocess.py")
