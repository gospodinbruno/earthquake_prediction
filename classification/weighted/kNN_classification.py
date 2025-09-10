import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
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


labels = ['0–1.9', '2.0–3.9', '4.0+']

df['Magnitude_Range'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False)


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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)


print("=== Multiclass Classification Report (kNN) ===")
print(classification_report(y_test, y_pred, digits=3))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (kNN - Multiclass)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


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
plt.title("Multiclass ROC Curves (kNN)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
