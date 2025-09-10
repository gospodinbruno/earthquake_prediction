import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

# Display class distribution
magnitude_labels = ['0–1.9', '2.0–3.9', '4.0+']
df['Magnitude_Range'] = pd.cut(df['Magnitude'], bins=bins, labels=magnitude_labels, right=False)
counts = df['Magnitude_Range'].value_counts().sort_index()
print("Earthquake Counts by Magnitude Range:")
for label, count in counts.items():
    print(f"{label}: {count}")

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
y = df['MagClass']

# Time-based split
cutoff = pd.Timestamp("2021-01-01")
X_train = X[df['Datetime'] < cutoff]
y_train = y[df['Datetime'] < cutoff]
X_test = X[df['Datetime'] >= cutoff]
y_test = y[df['Datetime'] >= cutoff]

# Standardize features (crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
print("\nTraining SVM Classifier...")
clf = SVC(
    kernel='rbf',  # Radial Basis Function kernel
    C=1.0,  # Regularization parameter
    gamma='scale',  # Kernel coefficient
    class_weight='balanced',  # Handle class imbalance
    probability=True,  # Enable probability estimates for ROC curves
    random_state=42
)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)

# Evaluation
print("\n=== Multiclass SVM Classification Report ===")
print(classification_report(y_test, y_pred, digits=3, target_names=['0–1.9', '2–3.9', '4+']))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (Multiclass SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Display SVM parameters
print(f"\n=== SVM Model Parameters ===")
print(f"Kernel: {clf.kernel}")
print(f"C (Regularization): {clf.C}")
print(f"Gamma: {clf.gamma}")
print(f"Number of Support Vectors: {clf.n_support_}")
print(f"Total Support Vectors: {sum(clf.n_support_)}")

# ROC Curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']
class_names = ['0–1.9', '2–3.9', '4+']

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {class_names[i]} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title("Multiclass ROC Curves (SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature analysis (using permutation importance since SVM doesn't have feature_importances_)
from sklearn.inspection import permutation_importance

print("\n=== Computing Permutation Feature Importance ===")
perm_importance = permutation_importance(clf, X_test_scaled, y_test, 
                                       n_repeats=10, random_state=42, n_jobs=-1)

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': features,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Feature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance_df)), importance_df['importance_mean'], 
         xerr=importance_df['importance_std'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (SVM - Multiclass)')
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

print(f"\n=== Summary ===")
print(f"Test Period: 2021-01-01 onwards")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(features)}")
print(f"Classes: 3 (0–1.9, 2–3.9, 4+ magnitude)")
