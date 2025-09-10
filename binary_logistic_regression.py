import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Load and sort data
df = pd.read_csv('earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Binary classification target
df['Target'] = (df['Magnitude'] >= 4.0).astype(int)

# Time features
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
y = df['Target']

# Time-based split
cutoff = pd.Timestamp("2021-01-01")
X_train = X[df['Datetime'] < cutoff]
y_train = y[df['Datetime'] < cutoff]
X_test = X[df['Datetime'] >= cutoff]
y_test = y[df['Datetime'] >= cutoff]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("=== Binary Classification Report ===")
print(classification_report(y_test, y_pred, digits=3))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['< 4.0', '≥ 4.0'],
            yticklabels=['< 4.0', '≥ 4.0'])
plt.title("Confusion Matrix (Binary Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (Binary Logistic Regression)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
