import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


def focal_loss(alpha=[0.1, 0.1, 1.0, 2.0], gamma=2.0):
    """
    Focal Loss implementation for multi-class classification.
    
    Args:
        alpha: List of weighting factors for each class [class0, class1, class2, class3]
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        ce = -y_true * tf.math.log(y_pred)
        
        p_t = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)
        
        alpha_t = tf.reduce_sum(alpha_tensor * y_true, axis=1, keepdims=True)
        
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        focal_loss = focal_weight * tf.reduce_sum(ce, axis=1, keepdims=True)
        
        return tf.squeeze(focal_loss, axis=1)
    
    return focal_loss_fixed


df = pd.read_csv('../../earthquake_2000_2021.csv', parse_dates=['Datetime'])
df.sort_values('Datetime', inplace=True)
df.reset_index(drop=True, inplace=True)


df = df[df['Magnitude'] >= 1.0].copy()
df.reset_index(drop=True, inplace=True)

bins = [1.0, 3.0, 5.0, 6.0, 10]
labels = [0, 1, 2, 3]
df['MagClass'] = pd.cut(df['Magnitude'], bins=bins, labels=labels, right=False).astype(int)


magnitude_labels = ['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+']
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

cutoff = pd.Timestamp("2017-01-01")
train_df = df[df['Datetime'] < cutoff].copy()
test_df = df[df['Datetime'] >= cutoff].copy()



original_class_weights = compute_class_weight('balanced', classes=np.unique(train_df['MagClass']), y=train_df['MagClass'])

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


y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax') 
])

focal_loss_fn = focal_loss(alpha=[0.1, 0.1, 1.0, 2.0], gamma=2.0)

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
print(classification_report(y_test, y_pred, digits=3, target_names=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+']))

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"Macro F1 Score:    {f1_macro:.3f}")
print(f"Weighted F1 Score: {f1_weighted:.3f}")

mcc_overall = matthews_corrcoef(y_test, y_pred)
print(f"MCC (overall, multiclass): {mcc_overall:.3f}")

classes = [0, 1, 2, 3]
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
            xticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+'],
            yticklabels=['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+'])
plt.title("Matrica zabune (Neuronska mreža)")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
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


y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
fpr, tpr, roc_auc = {}, {}, {}
colors = ['blue', 'orange', 'green', 'red']
class_names = ['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+']

plt.figure(figsize=(8, 6))
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Class {class_names[i]} ROC curve (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title("ROC krivulja (Neural Network)")
plt.xlabel("Stopa lažno pozitivnih")
plt.ylabel("Stopa istinsko pozitivnih")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


precision, recall, pr_auc = {}, {}, {}
colors = ['blue', 'orange', 'green', 'red']
class_names = ['1.0-2.9', '3.0-4.9', '5.0-5.9', '6.0+']

plt.figure(figsize=(8, 6))
for i in range(4):
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

print("\n=== Permutation Importance Analysis ===")

def model_predict_wrapper(X):
    """Wrapper function to make Keras model compatible with sklearn's permutation_importance"""
    predictions = model.predict(X, verbose=0)
    return np.argmax(predictions, axis=1)

print("Calculating permutation importance...")
perm_importance = permutation_importance(
    model_predict_wrapper, 
    X_test_scaled, 
    y_test, 
    n_repeats=10, 
    random_state=42,
    scoring='f1_macro'
)


feature_names = features
croatian_feature_names = {
    'Latitude': 'Geografska širina',
    'Longitude': 'Geografska dužina', 
    'Depth': 'Dubina',
    'hour_of_day': 'Sat dana',
    'day_of_week': 'Dan u tjednu',
    'time_since_last_eq': 'Vrijeme od zadnjeg potresa',
    'eq_count_last_24h': 'Broj potresa u zadnja 24h'
}

importance_df = pd.DataFrame({
    'feature': feature_names,
    'croatian_name': [croatian_feature_names[f] for f in feature_names],
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nRezultati važnosti permutacije (F1 Macro Score):")
print("=" * 60)
for idx, row in importance_df.iterrows():
    print(f"{row['croatian_name']:25s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(importance_df)), importance_df['importance_mean'], 
               yerr=importance_df['importance_std'], capsize=5, alpha=0.7)
plt.xlabel('Značajke')
plt.ylabel('Važnost permutacije')
plt.title('Važnost značajki NN modela')
plt.xticks(range(len(importance_df)), importance_df['croatian_name'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

for i, (bar, mean_val, std_val) in enumerate(zip(bars, importance_df['importance_mean'], importance_df['importance_std'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.001, 
             f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

plt.show()

print(f"\n=== Summary ===")
print(f"Test Period: 2020-01-01 onwards")
print(f"Training samples (original): {len(train_df)}")
print(f"Training samples (balanced): {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(features)}")
print(f"Classes: 4 (1.0-2.9, 3.0-4.9, 5.0-5.9, 6.0+ magnitude)")
print(f"Balancing method: class weights")
print(f"Model architecture: 64-64-4 with dropout")
