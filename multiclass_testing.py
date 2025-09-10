import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Load and sort data
df = pd.read_csv('earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Define magnitude classes
bins = [0, 2, 4, 10]
labels = [0, 1, 2]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)

# Feature engineering from time
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

# Time-based split: train before 2021, test 2021+
cutoff = pd.Timestamp("2021-01-01")
train_mask = df['Datetime'] < cutoff
test_mask = df['Datetime'] >= cutoff

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# -----------------------------
# 🔁 COMBINED RESAMPLING STRATEGY
# -----------------------------

# Convert to DataFrame for resampling
train_data = X_train.copy()
train_data['MagClass'] = y_train.values

# Separate by class
df_0 = train_data[train_data['MagClass'] == 0]
df_1 = train_data[train_data['MagClass'] == 1]
df_2 = train_data[train_data['MagClass'] == 2]

# Undersample class 0 and 1
df_0_down = resample(df_0, replace=False, n_samples=10000, random_state=42)
df_1_down = resample(df_1, replace=False, n_samples=10000, random_state=42)

# Combine and SMOTE class 2
combined_df = pd.concat([df_0_down, df_1_down, df_2])
X_combined = combined_df[features]
y_combined = combined_df['MagClass']

# Apply SMOTE to increase class 2
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

# -----------------------------
# 🔍 MODEL TRAINING & EVALUATION
# -----------------------------

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# F1 Scores
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {macro_f1:.3f}")

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)
importances.plot(kind='barh')
plt.title("Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# ROC Curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = clf.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(7, 5))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multiclass ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
