import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Load and sort data
df = pd.read_csv('../../earthquake.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Binary classification: Strong quake = magnitude ≥ 4.0
df['MagClass'] = (df['Magnitude'] >= 4.0).astype(int)

# Display class distribution
class_counts = df['MagClass'].value_counts().sort_index()
print("Earthquake Class Distribution:")
print(f"Weak earthquakes (< 4.0): {class_counts[0]}")
print(f"Strong earthquakes (≥ 4.0): {class_counts[1]}")
print(f"Class imbalance ratio: {class_counts[0] / class_counts[1]:.1f}:1")

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

# Standardize features (crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with balanced class weights
print("\nTraining SVM Classifier...")
clf = SVC(
    kernel='rbf',  # Radial Basis Function kernel
    C=1.0,  # Regularization parameter
    gamma='scale',  # Kernel coefficient
    class_weight='balanced',  # Handle class imbalance
    probability=True,  # Enable probability estimates
    random_state=42
)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

# Evaluation
print("\n=== Binary SVM Classification Report ===")
print("Classification Report (2021–2025):")
print(classification_report(y_test, y_pred, target_names=['Weak EQ', 'Strong EQ']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Weak EQ', 'Strong EQ'],
            yticklabels=['Weak EQ', 'Strong EQ'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Binary SVM, 2021–2025)')
plt.tight_layout()
plt.show()

# Display SVM parameters
print(f"\n=== SVM Model Parameters ===")
print(f"Kernel: {clf.kernel}")
print(f"C (Regularization): {clf.C}")
print(f"Gamma: {clf.gamma}")
print(f"Number of Support Vectors: {clf.n_support_}")
print(f"Total Support Vectors: {sum(clf.n_support_)}")

# Feature importance using permutation importance
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
plt.title('Feature Importance (SVM - Binary Classification)')
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'SVM ROC curve (AUC = {roc_auc:.3f})', color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (SVM Binary Classification)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Performance metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_class1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc_score_val = roc_auc_score(y_test, y_proba)

print(f"\n=== Performance Metrics ===")
print(f"F1 Score (macro): {f1_macro:.3f}")
print(f"F1 Score (strong earthquakes): {f1_class1:.3f}")
print(f"ROC AUC Score: {roc_auc_score_val:.3f}")

print(f"\n=== Summary ===")
print(f"Test Period: 2021-01-01 onwards")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(features)}")
print(f"Binary classification: Weak (< 4.0) vs Strong (≥ 4.0) earthquakes")
