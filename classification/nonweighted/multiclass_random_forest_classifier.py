import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


# Load and preprocess
df = pd.read_csv('../../earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Create multiclass labels: 0–2, 2–4, 4+
bins = [0, 2, 4, 10]
labels = [0, 1, 2]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)

labels = ['0–1.9', '2.0–3.9', '4.0+']

df['Magnitude_Range'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False)

# Count occurrences per bin
counts = df['Magnitude_Range'].value_counts().sort_index()

# Print the result
print("Earthquake Counts by Magnitude Range:")
for label, count in counts.items():
    print(f"{label}: {count}")

# Time-based features
df['hour_of_day'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['time_since_last_eq'] = df['Datetime'].diff().dt.total_seconds() / 3600
df['time_since_last_eq'].fillna(df['time_since_last_eq'].median(), inplace=True)

# Quakes in last 24h
eq_counts = []
for i in range(len(df)):
    start_time = df.loc[i, 'Datetime'] - timedelta(hours=24)
    count = df[(df['Datetime'] >= start_time) & (df['Datetime'] < df.loc[i, 'Datetime'])].shape[0]
    eq_counts.append(count)
df['eq_count_last_24h'] = eq_counts

# Feature set
features = ['Latitude', 'Longitude', 'Depth', 'hour_of_day', 'day_of_week',
            'time_since_last_eq', 'eq_count_last_24h']
X = df[features]
y = df['MagClass']

# Chronological split: train before 2021, test after
cutoff = pd.Timestamp("2021-01-01")
X_train = X[df['Datetime'] < cutoff]
y_train = y[df['Datetime'] < cutoff]
X_test = X[df['Datetime'] >= cutoff]
y_test = y[df['Datetime'] >= cutoff]

# Fit Random Forest (multiclass) with class weights
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report (2021–2025):")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–2', '2–4', '4+'],
            yticklabels=['0–2', '2–4', '4+'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Multiclass)')
plt.show()

# F1 scores
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {f1_macro:.3f}")

# Feature importance
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

importances.plot(kind='barh')
plt.title('Feature Importances (Random Forest – Multiclass)')
plt.gca().invert_yaxis()
plt.show()

# Multiclass ROC Curve
# Binarize the output
classes = [0, 1, 2]
y_test_bin = label_binarize(y_test, classes=classes)
y_score = clf.predict_proba(X_test)

# One-vs-rest ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(7, 5))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
