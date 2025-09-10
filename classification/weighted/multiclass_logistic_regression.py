import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Load and sort data
df = pd.read_csv('../../earthquake_2000_2021.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)


bins = [0, 2, 4, 10]
labels = [0, 1, 2]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)


magnitude_labels = ['0–1.9', '2.0–3.9', '4.0+']
df['Magnitude_Range'] = pd.cut(df['Magnitude'], bins=bins, labels=magnitude_labels, right=False)
counts = df['Magnitude_Range'].value_counts().sort_index()
print("Earthquake Counts by Magnitude Range:")
for label, count in counts.items():
    print(f"{label}: {count}")


df['hour_of_day'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['time_since_last_eq'] = df['Datetime'].diff().dt.total_seconds() / 3600
df['time_since_last_eq'].fillna(df['time_since_last_eq'].median(), inplace=True)


eq_counts = []
for i in range(len(df)):
    start_time = df.loc[i, 'Datetime'] - timedelta(hours=24)
    count = df[(df['Datetime'] >= start_time) & (df['Datetime'] < df.loc[i, 'Datetime'])].shape[0]
    eq_counts.append(count)
df['eq_count_last_24h'] = eq_counts


features = ['Latitude', 'Longitude', 'Depth', 'hour_of_day', 'day_of_week',
            'time_since_last_eq', 'eq_count_last_24h']
X = df[features]
y = df['MagClass']


train_start = pd.Timestamp("2000-01-01")
train_end = pd.Timestamp("2017-01-01")
train_df = df[(df['Datetime'] >= train_start) & (df['Datetime'] < train_end)].copy()
test_df = df[df['Datetime'] >= train_end].copy()

X_train = train_df[features]
y_train = train_df['MagClass']
X_test = test_df[features]
y_test = test_df['MagClass']

print(f"\nClass distribution in training data:")
unique, counts = np.unique(y_train, return_counts=True)
for class_label, count in zip(unique, counts):
    print(f"Class {class_label}: {count}")
print(f"Total training samples: {len(X_train)}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining Weighted Logistic Regression Classifier...")
clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)


coef_df = pd.DataFrame(clf.coef_, columns=features)
coef_df['Class'] = ['0–1.9', '2–3.9', '4+']
coef_df.set_index('Class', inplace=True)
print("\n=== Logistic Regression Coefficients (per class) ===")
print(coef_df.T)


coef_df.T.plot(kind='bar', figsize=(10, 6))
plt.title("Logistic Regression Coefficients by Feature and Class (Weighted)")
plt.ylabel("Coefficient Value")
plt.xlabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Magnitude Class")
plt.show()

# Feature importance using coefficient magnitudes (average across classes)
feature_importance = np.abs(coef_df).mean(axis=0).sort_values(ascending=False)
print("\nFeature Importances (Average Absolute Coefficients):")
print(feature_importance)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.title('Feature Importances (Weighted Logistic Regression)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()


print("\n=== Weighted Multiclass Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred, digits=3, target_names=['0–1.9', '2–3.9', '4+']))

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"Macro F1 Score:    {f1_macro:.3f}")
print(f"Weighted F1 Score: {f1_weighted:.3f}")

# ===== MCC (multiclass) =====
mcc_overall = matthews_corrcoef(y_test, y_pred)
print(f"MCC (overall, multiclass): {mcc_overall:.3f}")

classes = [0, 1, 2]
mcc_ovr = {}
for c in classes:
    y_true_bin = (y_test == c).astype(int)
    y_pred_bin = (y_pred == c).astype(int)
    mcc_ovr[c] = matthews_corrcoef(y_true_bin, y_pred_bin)
print("One-vs-Rest MCC per class:")
for c, m in mcc_ovr.items():
    print(f"  Class {magnitude_labels[c]}: {m:.3f}")


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (Weighted Multiclass Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


print(f"\n=== Logistic Regression Model Parameters ===")
print(f"Multi-class strategy: {clf.multi_class}")
print(f"Solver: {clf.solver}")
print(f"Max iterations: {clf.max_iter}")
print(f"Training samples (original): {len(train_df)}")
print(f"Training samples (balanced): {len(X_train)}")


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
plt.title("Multiclass ROC Curves (Weighted Logistic Regression)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


precision, recall, pr_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']
class_names = ['0–1.9', '2–3.9', '4+']

plt.figure(figsize=(8, 6))
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    pr_auc[i] = average_precision_score(y_test_bin[:, i], y_prob[:, i])
    plt.plot(recall[i], precision[i], color=colors[i],
             label=f'Class {class_names[i]} PR curve (AP = {pr_auc[i]:.2f})')


baseline = np.sum(y_test_bin) / len(y_test_bin)
plt.axhline(y=baseline, color='k', linestyle='--', label='Random Classifier')
plt.title("Multiclass Precision-Recall Curves (Weighted Logistic Regression)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\n=== Summary ===")
print(f"Test Period: 2020-01-01 onwards")
print(f"Training samples (original): {len(train_df)}")
print(f"Training samples (balanced): {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(features)}")
print(f"Classes: 3 (0–1.9, 2–3.9, 4+ magnitude)")
print(f"Balancing method: class_weight='balanced'")
print(f"Model: Multinomial Logistic Regression with L-BFGS solver")
