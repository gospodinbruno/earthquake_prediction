import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


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



original_class_weights = compute_class_weight('balanced', classes=np.unique(train_df['MagClass']), y=train_df['MagClass'])

# Apply square root to reduce extreme weights
conservative_weights = np.sqrt(original_class_weights)
class_weight_dict = {i: conservative_weights[i] for i in range(len(conservative_weights))}

print(f"\nOriginal class distribution in training data:")
print(f"Class 0: {len(train_df[train_df['MagClass'] == 0])}")
print(f"Class 1: {len(train_df[train_df['MagClass'] == 1])}")
print(f"Class 2: {len(train_df[train_df['MagClass'] == 2])}")
print(f"Original balanced weights: {original_class_weights}")
print(f"Conservative weights (sqrt): {class_weight_dict}")


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


y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# --- Neural Network Model ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax') 
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


print("\nTraining Weighted Neural Network...")
history = model.fit(
    X_train_scaled, y_train_cat, 
    epochs=25, 
    batch_size=64, 
    verbose=1, 
    validation_split=0.1,
    class_weight=class_weight_dict
)


y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n=== Classification Report (Weighted Neural Network - Multiclass) ===")
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
plt.title("Confusion Matrix (Weighted Neural Network)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']
class_names = ['0–1.9', '2–3.9', '4+']

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {class_names[i]} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title("Multiclass ROC Curves (Weighted Neural Network)")
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
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    pr_auc[i] = average_precision_score(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(recall[i], precision[i], color=colors[i],
             label=f'Class {class_names[i]} PR curve (AP = {pr_auc[i]:.2f})')


baseline = np.sum(y_test_bin) / len(y_test_bin)
plt.axhline(y=baseline, color='k', linestyle='--', label='Random Classifier')
plt.title("Multiclass Precision-Recall Curves (Weighted Neural Network)")
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
print(f"Balancing method: class weights")
print(f"Model architecture: 64-64-3 with dropout")
