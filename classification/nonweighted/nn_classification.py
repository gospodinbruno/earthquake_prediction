import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- Load and preprocess data ---
df = pd.read_csv('../../earthquake_2015_onwards_corrected.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Magnitude classes
bins = [0, 2, 4, 10]
labels = [0, 1, 2]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)

# Time-based features
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

# Chronological split
cutoff = pd.Timestamp("2020-01-01")
X_train = X[df['Datetime'] < cutoff]
y_train = y[df['Datetime'] < cutoff]
X_test = X[df['Datetime'] >= cutoff]
y_test = y[df['Datetime'] >= cutoff]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels
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

# Train model
history = model.fit(X_train_scaled, y_train_cat, epochs=25, batch_size=64, verbose=1, validation_split=0.1)

# Predict
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# --- Evaluation ---
print("=== Classification Report (Neural Network - Multiclass) ===")
print(classification_report(y_test, y_pred, digits=3))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0–1.9', '2–3.9', '4+'],
            yticklabels=['0–1.9', '2–3.9', '4+'])
plt.title("Confusion Matrix (Neural Network)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(7, 5))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multiclass ROC Curves (Neural Network)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Permutation Importance =====
print("\n=== Permutation Importance Analysis ===")

# Create a wrapper function for the Keras model to work with sklearn's permutation_importance
def model_predict_wrapper(X):
    """Wrapper function to make Keras model compatible with sklearn's permutation_importance"""
    predictions = model.predict(X, verbose=0)
    return np.argmax(predictions, axis=1)

# Calculate permutation importance
print("Calculating permutation importance...")
perm_importance = permutation_importance(
    model_predict_wrapper, 
    X_test_scaled, 
    y_test, 
    n_repeats=10, 
    random_state=42,
    scoring='f1_macro'
)

# Get feature names
feature_names = features

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Importance Results (F1 Macro Score):")
print("=" * 50)
for idx, row in importance_df.iterrows():
    print(f"{row['feature']:20s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

# Visualize permutation importance
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(importance_df)), importance_df['importance_mean'], 
               yerr=importance_df['importance_std'], capsize=5, alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Permutation Importance (F1 Macro Score)')
plt.title('Feature Importance - Neural Network Classification')
plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add value labels on bars
for i, (bar, mean_val, std_val) in enumerate(zip(bars, importance_df['importance_mean'], importance_df['importance_std'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.001, 
             f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

plt.show()
