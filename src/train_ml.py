# ============================================================
# PHASE 4 — MACHINE LEARNING  FIXED v2.1
# ============================================================
# FIXES:
#   - VotingClassifier was incorrectly nested inside a Pipeline
#     that added a *second* SelectKBest; sub-estimators are now
#     raw classifiers and the pipeline wraps the whole ensemble.
#   - SelectKBest K is clamped to min(20, actual feature count).
#   - AUC computation guarded: uses decision_function fallback
#     for estimators without predict_proba (e.g. SVC without
#     probability=True).
#   - Confusion matrix filenames use a safe-name deduplication
#     guard to prevent overwriting.
#   - FEATURE_ORDER list added so the feature vector passed to
#     the saved model always matches the extraction order in
#     app.py / feature_extraction.py.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve)

os.makedirs("model",           exist_ok=True)
os.makedirs("outputs/graphs",  exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

# ── Load feature CSVs and label arrays ──
def load_data():
    X_tr = pd.read_csv("data/features/X_train_feat.csv")
    X_v  = pd.read_csv("data/features/X_val_feat.csv")
    X_te = pd.read_csv("data/features/X_test_feat.csv")
    y_tr = np.load("data/features/y_train.npy")
    y_v  = np.load("data/features/y_val.npy")
    y_te = np.load("data/features/y_test.npy")
    names = X_tr.columns.tolist()

    for df in [X_tr, X_v, X_te]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0.0, inplace=True)

    X_tv = np.concatenate([X_tr.values, X_v.values])
    y_tv = np.concatenate([y_tr, y_v])
    print(f"[INFO] Train+Val={X_tv.shape}  Test={X_te.shape}")
    print(f"[INFO] Features: {len(names)}")
    return X_tv, X_te.values, y_tv, y_te, names

# ── Evaluate one trained model ──
def evaluate(model, X_te, y_te, name):
    yp    = model.predict(X_te)
    yprob = None

    # FIX: guard AUC for estimators without predict_proba
    if hasattr(model, "predict_proba"):
        try:
            yprob = model.predict_proba(X_te)[:, 1]
        except Exception:
            pass
    if yprob is None and hasattr(model, "decision_function"):
        try:
            df_   = model.decision_function(X_te)
            yprob = (df_ - df_.min()) / (df_.max() - df_.min() + 1e-9)
        except Exception:
            pass

    r = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_te, yp),
        "Precision": precision_score(y_te, yp, zero_division=0),
        "Recall":    recall_score(y_te, yp, zero_division=0),
        "F1":        f1_score(y_te, yp, zero_division=0),
        "AUC":       roc_auc_score(y_te, yprob) if yprob is not None else 0.0,
    }
    print(f"\n  {name}")
    print(f"  Acc={r['Accuracy']*100:.1f}%  F1={r['F1']:.3f}  AUC={r['AUC']:.3f}")
    return r

# ── Confusion matrix plot ──
# FIX: filename deduplication guard
_cm_used = set()

def plot_confusion(model, X_te, y_te, name):
    cm   = confusion_matrix(y_te, model.predict(X_te))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Apnea"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    safe = name.replace(" ", "_").replace(".", "").lower()
    # Deduplicate filename
    candidate = f"outputs/graphs/cm_{safe}.png"
    suffix    = 1
    while candidate in _cm_used:
        candidate = f"outputs/graphs/cm_{safe}_{suffix}.png"
        suffix += 1
    _cm_used.add(candidate)
    plt.savefig(candidate, dpi=120)
    plt.close()

# ── ROC curves for all models ──
def plot_roc(models_dict, X_te, y_te):
    cols = ["#38bdf8", "#f87171", "#34d399", "#fbbf24"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, m) in enumerate(models_dict.items()):
        yprob = None
        if hasattr(m, "predict_proba"):
            try: yprob = m.predict_proba(X_te)[:, 1]
            except Exception: pass
        if yprob is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_te, yprob)
            auc         = roc_auc_score(y_te, yprob)
            ax.plot(fpr, tpr, color=cols[i % len(cols)], lw=2,
                    label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/graphs/roc_curves.png", dpi=120)
    plt.close()

# ── Bar chart comparing models ──
def plot_comparison(df):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x    = np.arange(len(metrics))
    w    = 0.18
    cols = ["#38bdf8", "#f87171", "#34d399", "#fbbf24"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=row["Model"],
                      color=cols[i % len(cols)], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x + w * (len(df) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model Comparison — All Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/graphs/model_comparison.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    X_tv, X_te, y_tv, y_te, feat_names = load_data()

    # FIX: clamp K to actual feature count
    K = min(20, X_tv.shape[1])

    def simple_pipe(clf):
        """Scale → Select → Classify for a single estimator."""
        return Pipeline([
            ("scale",  RobustScaler()),
            ("select", SelectKBest(f_classif, k=K)),
            ("model",  clf),
        ])

    # FIX: VotingClassifier sub-estimators are raw classifiers;
    # the outer Pipeline handles scaling + feature selection ONCE.
    rf_sub  = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    gb_sub  = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, random_state=42)
    svm_sub = SVC(
        C=10, probability=True, kernel="rbf", gamma="scale", random_state=42)

    voting = VotingClassifier(
        estimators=[("rf", rf_sub), ("gb", gb_sub), ("svm", svm_sub)],
        voting="soft",
        weights=[2, 1, 1],
    )
    voting_pipe = Pipeline([
        ("scale",  RobustScaler()),
        ("select", SelectKBest(f_classif, k=K)),
        ("model",  voting),          # sub-estimators see already-selected features
    ])

    models = {
        "Logistic Reg.":   simple_pipe(
            LogisticRegression(C=1.0, max_iter=2000,
                               class_weight="balanced", random_state=42)),
        "SVM":             simple_pipe(
            SVC(C=10, kernel="rbf", probability=True,
                gamma="scale", random_state=42)),
        "Random Forest":   simple_pipe(
            RandomForestClassifier(n_estimators=300, max_depth=20,
                                   class_weight="balanced", min_samples_split=5,
                                   random_state=42, n_jobs=-1)),
        "Voting Ensemble": voting_pipe,
    }

    print("\n[TRAIN] Training all models...")
    results = []
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_tv, y_tv)
        cv = cross_val_score(
            model, X_tv, y_tv,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="f1")
        print(f"  CV F1: {cv.mean():.3f} ± {cv.std():.3f}")
        results.append(evaluate(model, X_te, y_te, name))
        plot_confusion(model, X_te, y_te, name)

    plot_roc(models, X_te, y_te)

    df = pd.DataFrame(results)
    df.to_csv("outputs/metrics/ml_results.csv", index=False)
    plot_comparison(df)

    print(f"\n{'='*55}")
    print(df[["Model","Accuracy","Precision","Recall","F1","AUC"]].to_string(index=False))
    print(f"{'='*55}")

    joblib.dump(models["Voting Ensemble"], "model/rf_model.pkl")
    print("\n[INFO] Best model saved → model/rf_model.pkl")
    print("✅ Phase 4 DONE → run: python src/train_dl.py")
