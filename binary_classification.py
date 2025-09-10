# binary_earthquake_classifier.py
# Binary target: >=4.0 (positive) vs <4.0 (negative)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# 1) Load & preprocess
# -----------------------------------------------------------------------------
df = pd.read_csv('earthquake.csv', parse_dates=['Datetime'])

# Sort chronologically and index by time for time-based rolling ops
df = df.sort_values('Datetime').reset_index(drop=True)
df = df.set_index('Datetime')

# Binary label: >=4.0 -> 1, else 0
df['y'] = (df['Magnitude'] >= 4.0).astype(int)

# Time-based features
df['hour_of_day'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['time_since_last_eq'] = df.index.to_series().diff().dt.total_seconds() / 3600.0
df['time_since_last_eq'] = df['time_since_last_eq'].fillna(df['time_since_last_eq'].median())

# Efficient quake count in the previous 24 hours (exclude current event)
# rolling('24H').count() includes current row; shift(1) to count strictly before now
df['eq_count_last_24h'] = (
    df['Magnitude']
    .rolling('24H')
    .count()
    .shift(1)
    .fillna(0)
    .astype(int)
)

# Final feature set
features = [
    'Latitude', 'Longitude', 'Depth',
    'hour_of_day', 'day_of_week',
    'time_since_last_eq', 'eq_count_last_24h'
]

# Keep rows with all needed columns
df = df.dropna(subset=features + ['y'])

# -----------------------------------------------------------------------------
# 2) Chronological split (no leakage)
# -----------------------------------------------------------------------------
cutoff = pd.Timestamp("2021-01-01")
train_df = df[df.index < cutoff].copy()
test_df  = df[df.index >= cutoff].copy()

X_train = train_df[features].copy()
y_train = train_df['y'].copy()
X_test  = test_df[features].copy()
y_test  = test_df['y'].copy()

print("Class balance (TRAIN):")
print(y_train.value_counts().rename({0:"<4.0",1:"≥4.0"}))
print("\nClass balance (TEST):")
print(y_test.value_counts().rename({0:"<4.0",1:"≥4.0"}))
print()

# -----------------------------------------------------------------------------
# 3) Scaling (needed for LR; trees ignore scaling)
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 4) Models
#    - Logistic Regression (class_weight='balanced')
#    - Random Forest (class_weight='balanced')
# -----------------------------------------------------------------------------
models = {
    "LogReg": LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    ),
    "RandForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ),
}

# Fit: scale for LR, raw for RF
models_fitted = {}
for name, model in models.items():
    if name == "LogReg":
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    models_fitted[name] = model

# -----------------------------------------------------------------------------
# 5) Evaluation helpers
# -----------------------------------------------------------------------------
def evaluate_binary(model, name, X_tr, X_te, scaled=False):
    if scaled:
        X_tr_use, X_te_use = X_train_scaled, X_test_scaled
    else:
        X_tr_use, X_te_use = X_train, X_test

    # Probabilities and default threshold 0.5
    y_prob = model.predict_proba(X_te_use)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"=== {name} @ threshold=0.50 ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<4.0','≥4.0'], yticklabels=['<4.0','≥4.0'])
    plt.title(f'Confusion Matrix ({name}, thr=0.50)')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.show()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4.8, 4.0))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve ({name})'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout(); plt.show()

    # Precision-Recall (better for rare events)
    precision, recall, thr = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(4.8, 4.0))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.title(f'Precision-Recall ({name})'); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc='lower left'); plt.grid(True); plt.tight_layout(); plt.show()

    # Threshold sweep → maximize F1 for positive class
    best_f1, best_thr = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        yp = (y_prob >= t).astype(int)
        f1_pos = f1_score(y_test, yp, pos_label=1)
        if f1_pos > best_f1:
            best_f1, best_thr = f1_pos, t

    print(f"Best positive-class F1 over thresholds: {best_f1:.3f} at thr={best_thr:.2f}")
    y_pred_best = (y_prob >= best_thr).astype(int)
    print(f"=== {name} @ threshold={best_thr:.2f} (chosen for F1_pos) ===")
    print(classification_report(y_test, y_pred_best, digits=3))
    print()

    # Return for optional downstream use
    return {
        "y_prob": y_prob,
        "best_thr": best_thr,
        "best_f1_pos": best_f1
    }

# -----------------------------------------------------------------------------
# 6) Run evaluations
# -----------------------------------------------------------------------------
res_lr  = evaluate_binary(models_fitted["LogReg"], "Logistic Regression",
                          X_train, X_test, scaled=True)

res_rf  = evaluate_binary(models_fitted["RandForest"], "Random Forest",
                          X_train, X_test, scaled=False)

# -----------------------------------------------------------------------------
# 7) (Optional) Feature importances for RF
# -----------------------------------------------------------------------------
rf = models_fitted["RandForest"]
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(6, 3.6))
importances.plot(kind='barh')
plt.title('Feature Importances (Random Forest)')
plt.tight_layout(); plt.show()

print("\nTop RF importances:\n", importances.sort_values(ascending=False))
