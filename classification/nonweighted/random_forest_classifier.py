import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score

# Optional: SMOTE for oversampling
# from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('../../earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Binary classification: Strong quake = magnitude ≥ 4.0
df['MagClass'] = (df['Magnitude'] >= 4.0).astype(int)

# Time-based features
df['hour_of_day'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['time_since_last_eq'] = df['Datetime'].diff().dt.total_seconds() / 3600
df['time_since_last_eq'].fillna(df['time_since_last_eq'].median(), inplace=True)

# Number of quakes in last 24h
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

# Train on data before 2021, test on 2021 and later
cutoff_date = pd.Timestamp("2021-01-01")
X_train = X[df['Datetime'] < cutoff_date]
y_train = y[df['Datetime'] < cutoff_date]
X_test = X[df['Datetime'] >= cutoff_date]
y_test = y[df['Datetime'] >= cutoff_date]

 # Optional: SMOTE to balance training data
#smote = SMOTE(random_state=42)
#X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Random Forest with class_weight
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report (2021–2025):")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (2021–2025)')
plt.show()

# Feature importances
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

importances.plot(kind='barh')
plt.title('Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.show()



y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Random Forest, Class 1 = Strong EQ)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# F1 Score
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_class1 = f1_score(y_test, y_pred, pos_label=1)

print(f"F1 Score (macro): {f1_macro:.3f}")
print(f"F1 Score (class 1 - strong earthquakes): {f1_class1:.3f}")
print(f"ROC AUC Score: {roc_auc:.3f}")