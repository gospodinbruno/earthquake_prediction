import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# ----------------------------
# 1) Load & basic preprocessing
# ----------------------------
df = pd.read_csv('../../earthquake_2000_2021.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Filter out earthquakes with magnitude < 1.0
df = df[df['Magnitude'] >= 1.0].copy()
df.reset_index(drop=True, inplace=True)

# Define bins (4-class setup): [1.0, 3.0), [3.0, 5.0), [5.0, 6.0), [6.0, 10)
bins = [1.0, 3.0, 5.0, 6.0, 10]
class_labels = [0, 1, 2, 3]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=class_labels, right=False).astype(int)

range_labels = ['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+']
df['Magnitude_Range'] = pd.cut(df['Magnitude'], bins=bins, labels=range_labels, right=False)

print("Earthquake Counts by Magnitude Range (>= 1.0):")
counts = df['Magnitude_Range'].value_counts().sort_index()
for label, count in counts.items():
    print(f"{label}: {count}")

print(f"\nTotal earthquakes after filtering (>= 1.0): {len(df)}")
print(f"Magnitude range: {df['Magnitude'].min():.2f} - {df['Magnitude'].max():.2f}")

# Time features (unchanged)
df['hour_of_day'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['time_since_last_eq'] = df['Datetime'].diff().dt.total_seconds() / 3600
df['time_since_last_eq'].fillna(df['time_since_last_eq'].median(), inplace=True)

# Rolling count of events in the last 24h (naive loop; OK for clarity)
eq_counts = []
for i in range(len(df)):
    start_time = df.loc[i, 'Datetime'] - timedelta(hours=24)
    count = df[(df['Datetime'] >= start_time) & (df['Datetime'] < df.loc[i, 'Datetime'])].shape[0]
    eq_counts.append(count)
df['eq_count_last_24h'] = eq_counts

# Features
features = [
    'Latitude', 'Longitude', 'Depth',
    'hour_of_day', 'day_of_week',
    'time_since_last_eq', 'eq_count_last_24h'
]
X = df[features]
y = df['MagClass']

# ----------------------------
# 2) Chronological split (unchanged)
# ----------------------------
cutoff = pd.Timestamp("2017-01-01")
train_df = df[df['Datetime'] < cutoff].copy()
test_df  = df[df['Datetime'] >= cutoff].copy()

X_train = train_df[features]
y_train = train_df['MagClass']
X_test  = test_df[features]
y_test  = test_df['MagClass']

print(f"\nOriginal class distribution in training data:")
print(f"Class 0: {len(y_train[y_train == 0])}")
print(f"Class 1: {len(y_train[y_train == 1])}")
print(f"Class 2: {len(y_train[y_train == 2])}")
print(f"Class 3: {len(y_train[y_train == 3])}")

# ----------------------------
# 3) Scaling (unchanged)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ----------------------------
# 4) Train kNN with distance weighting
# ----------------------------
clf = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
clf.fit(X_train_scaled, y_train)

# Vanilla predictions
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)  # shape (n_samples, n_classes)

# ----------------------------
# 5) Prior-corrected probabilities (closest to class_weight='balanced')
#    - weights = inverse class priors from TRAIN
#    - dampen with sqrt to avoid overcorrection
# ----------------------------
classes = clf.classes_  # ordered as in predict_proba columns
class_counts = np.array([(y_train == c).sum() for c in classes], dtype=float)
priors = class_counts / class_counts.sum()

inv_prior = 1.0 / priors
inv_prior /= inv_prior.mean()   # center around 1.0 for stability
inv_prior = np.sqrt(inv_prior)  # sqrt dampening

# Apply to predicted probabilities and renormalize
y_prob_w = y_prob * inv_prior
row_sums = y_prob_w.sum(axis=1, keepdims=True)
# Guard against potential numerical issues
row_sums[row_sums == 0] = 1.0
y_prob_w = y_prob_w / row_sums

# Weighted predictions
y_pred_w = classes[y_prob_w.argmax(axis=1)]

# ----------------------------
# 6) Reports & metrics
# ----------------------------
print("\n=== kNN (weights='distance') — Vanilla ===")
print(classification_report(y_test, y_pred, digits=3, target_names=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+']))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print("Weighted F1:", f1_score(y_test, y_pred, average='weighted'))
print("MCC:", matthews_corrcoef(y_test, y_pred))

print("\n=== kNN + Prior-Corrected Probs (√-dampened) ===")
print(classification_report(y_test, y_pred_w, digits=3, target_names=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+']))
print("Macro F1 (weighted probs):", f1_score(y_test, y_pred_w, average='macro'))
print("Weighted F1 (weighted probs):", f1_score(y_test, y_pred_w, average='weighted'))
print("MCC (weighted probs):", matthews_corrcoef(y_test, y_pred_w))

# Confusion matrices
cm_v = confusion_matrix(y_test, y_pred, labels=classes)
cm_w = confusion_matrix(y_test, y_pred_w, labels=classes)

plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
sns.heatmap(cm_v, annot=True, fmt='d', cmap='Blues',
            xticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+'],
            yticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+'])
plt.title("Confusion Matrix — Vanilla kNN")
plt.xlabel("Predicted"); plt.ylabel("Actual")

plt.subplot(1,2,2)
sns.heatmap(cm_w, annot=True, fmt='d', cmap='Greens',
            xticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+'],
            yticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+'])
plt.title("Confusion Matrix — Prior-Corrected (√)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ----------------------------
# 7) ROC curves (One-vs-Rest) using prior-corrected probabilities
# ----------------------------
y_test_bin = label_binarize(y_test, classes=classes)
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red']
class_names = ['1.0-2.9', '3.0-4.9', '5.0-5.9', '6+']
for i, c in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_w[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
             label=f'Class {class_names[i]} ROC (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title("Multiclass ROC (kNN, Prior-Corrected Probs)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
