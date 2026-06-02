import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Import deep learning components
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# 1. Data Preparation & Scaling
# ==========================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Requirement: Scale data (StandardScaler) before passing it to the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 2. Train Model A (XGBoost - Classical ML)
# ==========================================
print("Training Model A (XGBoost)...")
model_a = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
model_a.fit(X_train_scaled, y_train)

# Evaluate Model A
y_pred_a = model_a.predict(X_test_scaled)
accuracy_a = float(accuracy_score(y_test, y_pred_a))
f1_a = float(f1_score(y_test, y_pred_a))

print(f"-> Model A Results - Accuracy: {accuracy_a:.4f}, F1-Score: {f1_a:.4f}\n")


# ==========================================
# 3. Train Model B (FNN - Deep Learning)
# ==========================================
print("Training Model B (Feedforward Neural Network)...")

# Decided depth: A 3-layer deep neural network (2 hidden layers + 1 output layer)
model_b = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Hidden Layer 1
    Dense(16, activation='relu'),                                         # Hidden Layer 2
    Dense(1, activation='sigmoid')                                        # Output Layer (Binary)
])

model_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
model_b.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)

# Evaluate Model B
loss_b, accuracy_b = model_b.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_b_prob = model_b.predict(X_test_scaled, verbose=0)
y_pred_b = (y_pred_b_prob > 0.5).astype(int).flatten()
f1_b = float(f1_score(y_test, y_pred_b))
accuracy_b = float(accuracy_b)

print(f"-> Model B Results - Accuracy: {accuracy_b:.4f}, F1-Score: {f1_b:.4f}\n")


# ==========================================
# 4. MLOps Automation & Artifact Retention
# ==========================================
print("--- Running Pipeline Model Comparison ---")

# Enforce Artifact Retention: Clear old files to ensure ONLY the winning model is saved
for f in ['best_model.pkl', 'best_model.h5']:
    if os.path.exists(f):
        os.remove(f)

# Automatic comparison via if/else statement
if accuracy_a >= accuracy_b:
    print("🏆 Winner: Model A (Classical ML - XGBoost)")
    
    # Save Classical ML model as .pkl
    joblib.dump(model_a, 'best_model.pkl')
    
    winning_metrics = {
        "winning_model": "Classical",
        "algorithm": "XGBoost",
        "accuracy": accuracy_a,
        "f1_score": f1_a
    }
else:
    print("🏆 Winner: Model B (Deep Learning - FNN)")
    
    # Save Deep Learning model as .h5
    model_b.save('best_model.h5')
    
    winning_metrics = {
        "winning_model": "DeepLearning",
        "algorithm": "Feedforward Neural Network",
        "accuracy": accuracy_b,
        "f1_score": f1_b
    }

# Save the winning model's metrics to metrics.json
with open('metrics.json', 'w') as f:
    json.dump(winning_metrics, f, indent=4)

print("✅ MLOps tracking complete. metrics.json generated and winning artifact retained successfully.")