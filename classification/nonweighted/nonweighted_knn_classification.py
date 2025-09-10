import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Load and sort data
df = pd.read_csv('../../earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Define magnitude classes
bins = [0, 2, 4, 10]
labels = [0, 1, 2]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)

# Feature engineering
df['hour_of_day'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['time_since_last_eq'] = df['Datetime'].diff().dt.total_seconds() / 3600
df['time_since_last_eq'].fillna(df['time_since_last_eq'].median(), inplace=True)

# Earthquakes in past 24h
eq_counts = []
for i in range(len(df)):
    start_time = df.loc[i, 'Datetime'] - timedelta(hours=24)
    count = df[(df['Datetime'] >= start_time) & (df['Datetime'] < df.loc[i, 'Datetime'])].shape[0]
    eq_counts.append(count)
df['eq_count_last_24h'] = eq_counts

# Features and labels
features = ['Latitude', 'Longitude', 'Depth', 'hour_of_day', 'day_of_week',
            'time_since_last_eq', 'eq_count_last_24h']
X = df[features]
y = df['MagClass']

# Time-based split
cutoff = pd.Timestamp("2021-01-01")
X_train = X[df['Datetime'] < cutoff]
y_train = y[df['Datetime'] < cutoff]
X_test = X[df['Datetime'] >= cutoff]
y_test = y[df['Datetime'] >= cutoff]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train kNN classifier (no balancing, no weights)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)

# Evaluation
print("=== Multiclass Classification Report (kNN, No Balancing) ===")
print(classification_report(y_test, y_pred, digits=3))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (kNN - No Resampling)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(7, 5))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multiclass ROC Curves (kNN - No Resampling)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
