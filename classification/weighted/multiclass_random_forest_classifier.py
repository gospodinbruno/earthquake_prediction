import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


df = pd.read_csv('../../earthquake_2015_onwards_corrected.csv', parse_dates=['Datetime'])
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

cutoff = pd.Timestamp("2020-01-01")
train_df = df[df['Datetime'] < cutoff].copy()
test_df = df[df['Datetime'] >= cutoff].copy()

X_train = train_df[features]
y_train = train_df['MagClass']
X_test = test_df[features]
y_test = test_df['MagClass']

print(f"\nOriginal class distribution in training data:")
print(f"Class 0: {len(y_train[y_train == 0])}")
print(f"Class 1: {len(y_train[y_train == 1])}")
print(f"Class 2: {len(y_train[y_train == 2])}")

majority_class_count = max(np.bincount(y_train))
target_counts = {
    0: majority_class_count,
    1: int(majority_class_count * 0.7),
    2: int(majority_class_count * 0.3)
}

smote = SMOTE(sampling_strategy=target_counts, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nClass distribution after conservative SMOTE:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for class_label, count in zip(unique, counts):
    print(f"Class {class_label}: {count}")
print(f"Total balanced training samples: {len(X_train_balanced)}")


X_train = X_train_balanced
y_train = y_train_balanced

print("\nTraining Weighted Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced',
    random_state=42,
    max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=2
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("\n=== Weighted Multiclass Random Forest Classification Report ===")
print("Classification Report (2020–2021):")
print(classification_report(y_test, y_pred, digits=3, target_names=['0–1.9', '2–3.9', '4+']))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Weighted Multiclass Random Forest)')
plt.tight_layout()
plt.show()


f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {f1_macro:.3f}")


importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title('Feature Importances (Weighted Random Forest – Multiclass)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()


print(f"\n=== Random Forest Model Parameters ===")
print(f"Number of estimators: {clf.n_estimators}")
print(f"Max depth: {clf.max_depth}")
print(f"Min samples split: {clf.min_samples_split}")
print(f"Min samples leaf: {clf.min_samples_leaf}")


# Binarize the output
classes = [0, 1, 2]
y_test_bin = label_binarize(y_test, classes=classes)
y_score = clf.predict_proba(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green']
class_names = ['0–1.9', '2–3.9', '4+']
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {class_names[i]} ROC curve (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves (Weighted Random Forest)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


precision, recall, pr_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']
class_names = ['0–1.9', '2–3.9', '4+']

plt.figure(figsize=(8, 6))
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall[i], precision[i], color=colors[i],
             label=f'Class {class_names[i]} PR curve (AP = {pr_auc[i]:.2f})')


baseline = np.sum(y_test_bin) / len(y_test_bin)
plt.axhline(y=baseline, color='k', linestyle='--', label='Random Classifier')
plt.title("Multiclass Precision-Recall Curves (Weighted Random Forest)")
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
print(f"Balancing method: SMOTE + class_weight='balanced'")
print(f"Model: Random Forest with 100 estimators")
