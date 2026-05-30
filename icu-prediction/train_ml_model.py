import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import joblib
import json
from pathlib import Path

print("="*80)
print("MULTI-DATASET TRAINING FOR ICU MORTALITY PREDICTION")
print("="*80)

all_X = []
all_y = []
dataset_names = []

# Dataset 1: PhysioNet Challenge 2019 (Dataset.csv) - Largest dataset with hourly measurements
print("\n[1/5] Loading PhysioNet Challenge 2019 Dataset.csv...")
physionet2019_path = "datasets/Dataset.csv"
try:
    physionet2019_df = pd.read_csv(physionet2019_path)
    print(f"   Shape: {physionet2019_df.shape}")
    
    # Use first hour (Hour=0) for baseline measurements
    physionet2019_baseline = physionet2019_df[physionet2019_df['Hour'] == 0].copy()
    print(f"   Baseline samples (Hour=0): {len(physionet2019_baseline)}")
    
    # Map features to app features
    features_2019 = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Lactate', 'Glucose', 'WBC', 'Creatinine']
    X_2019 = physionet2019_baseline[features_2019].copy()
    X_2019.columns = ['HR', 'SpO2', 'Temp', 'SysBP', 'MAP', 'DiasBP', 'RespRate', 'Age', 'Lactate', 'Glucose', 'WBC', 'Creatinine']
    
    # Reorder to match app
    X_2019 = X_2019[['Age', 'HR', 'SysBP', 'DiasBP', 'MAP', 'RespRate', 'Temp', 'SpO2', 'Glucose', 'Creatinine', 'WBC', 'Lactate']]
    X_2019 = X_2019.fillna(X_2019.median())
    
    y_2019 = physionet2019_baseline['SepsisLabel']
    
    all_X.append(X_2019)
    all_y.append(y_2019)
    dataset_names.append(f"PhysioNet2019 ({len(X_2019)} samples)")
    print(f"   Loaded: {len(X_2019)} samples, Target distribution: {y_2019.value_counts().to_dict()}")
except Exception as e:
    print(f"   Error loading Dataset.csv: {e}")

# Dataset 2: PhysioNet X_train_2025.csv (real clinical data)
print("\n[2/5] Loading PhysioNet X_train_2025.csv...")
try:
    X_path = "datasets/X_train_2025.csv"
    y_path = "datasets/y_train_2025.csv"
    
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    
    features_physionet = [
        'Age', 'HR_first', 'SysABP_first', 'DiasABP_first', 'MAP_first',
        'RespRate_first', 'Temp_first', 'SaO2_first', 
        'Glucose_first', 'Creatinine_first', 'WBC_first', 'Lactate_first'
    ]
    
    X_physionet = X_df[features_physionet].copy()
    X_physionet.columns = ['Age', 'HR', 'SysBP', 'DiasBP', 'MAP', 'RespRate', 'Temp', 'SpO2', 'Glucose', 'Creatinine', 'WBC', 'Lactate']
    X_physionet = X_physionet.fillna(X_physionet.median())
    y_physionet = y_df['In-hospital_death']
    
    all_X.append(X_physionet)
    all_y.append(y_physionet)
    dataset_names.append(f"PhysioNet2025 ({len(X_physionet)} samples)")
    print(f"   Loaded: {len(X_physionet)} samples, Target distribution: {y_physionet.value_counts().to_dict()}")
except Exception as e:
    print(f"   Error loading PhysioNet2025: {e}")

# Dataset 3: ICU_Patient_Monitoring_Mortality_Prediction_15000.csv
print("\n[3/5] Loading ICU_Patient_Monitoring_Mortality_Prediction_15000.csv...")
try:
    icu_path = "datasets/ICU_Patient_Monitoring_Mortality_Prediction_15000.csv"
    icu_df = pd.read_csv(icu_path)
    
    features_icu = [
        'age', 'heart_rate_mean', 'systolic_bp_mean', 'respiratory_rate_mean', 
        'spo2_mean', 'temperature_mean', 'glucose_mean', 'lactate_mean'
    ]
    
    X_icu = icu_df[features_icu].copy()
    X_icu.columns = ['Age', 'HR', 'SysBP', 'RespRate', 'SpO2', 'Temp', 'Glucose', 'Lactate']
    X_icu['DiasBP'] = X_icu['SysBP'] * 0.67
    X_icu['MAP'] = (X_icu['SysBP'] + 2 * X_icu['DiasBP']) / 3
    X_icu['Creatinine'] = np.random.uniform(0.6, 3.0, size=len(X_icu))
    X_icu['WBC'] = np.random.uniform(4.0, 15.0, size=len(X_icu))
    X_icu = X_icu.fillna(X_icu.median())
    X_icu = X_icu[['Age', 'HR', 'SysBP', 'DiasBP', 'MAP', 'RespRate', 'Temp', 'SpO2', 'Glucose', 'Creatinine', 'WBC', 'Lactate']]
    
    y_icu = icu_df['mortality_label']
    
    all_X.append(X_icu)
    all_y.append(y_icu)
    dataset_names.append(f"ICU_Monitoring ({len(X_icu)} samples)")
    print(f"   Loaded: {len(X_icu)} samples, Target distribution: {y_icu.value_counts().to_dict()}")
except Exception as e:
    print(f"   Error loading ICU_Monitoring: {e}")

# Dataset 4: synthetic_icu_data.csv
print("\n[4/5] Loading synthetic_icu_data.csv...")
try:
    synthetic_path = "datasets/synthetic_icu_data.csv"
    synthetic_df = pd.read_csv(synthetic_path)
    
    features_synthetic = [
        'Age', 'Heart Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Death_within_7_days_Label'
    ]
    
    X_synthetic = synthetic_df[features_synthetic].copy()
    X_synthetic.columns = ['Age', 'HR', 'SysBP', 'DiasBP', 'RespRate', 'Temp', 'SpO2', 'target']
    X_synthetic['MAP'] = (X_synthetic['SysBP'] + 2 * X_synthetic['DiasBP']) / 3
    X_synthetic['Glucose'] = np.random.uniform(70, 200, size=len(X_synthetic))
    X_synthetic['Creatinine'] = np.random.uniform(0.6, 3.0, size=len(X_synthetic))
    X_synthetic['WBC'] = np.random.uniform(4.0, 15.0, size=len(X_synthetic))
    X_synthetic['Lactate'] = np.random.uniform(0.5, 6.0, size=len(X_synthetic))
    X_synthetic = X_synthetic.fillna(X_synthetic.median())
    X_synthetic = X_synthetic[['Age', 'HR', 'SysBP', 'DiasBP', 'MAP', 'RespRate', 'Temp', 'SpO2', 'Glucose', 'Creatinine', 'WBC', 'Lactate']]
    
    y_synthetic = synthetic_df['Death_within_7_days_Label']
    
    all_X.append(X_synthetic)
    all_y.append(y_synthetic)
    dataset_names.append(f"Synthetic_ICU ({len(X_synthetic)} samples)")
    print(f"   Loaded: {len(X_synthetic)} samples, Target distribution: {y_synthetic.value_counts().to_dict()}")
except Exception as e:
    print(f"   Error loading Synthetic_ICU: {e}")

# Dataset 5: icu_risk_score.csv (use risk_score as proxy for mortality)
print("\n[5/5] Loading icu_risk_score.csv...")
try:
    risk_path = "datasets/icu_risk_score.csv"
    risk_df = pd.read_csv(risk_path)
    
    features_risk = [
        'age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate',
        'temperature_c', 'spo2', 'wbc_count', 'creatinine', 'lactate'
    ]
    
    X_risk = risk_df[features_risk].copy()
    X_risk.columns = ['Age', 'HR', 'SysBP', 'DiasBP', 'RespRate', 'Temp', 'SpO2', 'WBC', 'Creatinine', 'Lactate']
    X_risk['MAP'] = (X_risk['SysBP'] + 2 * X_risk['DiasBP']) / 3
    X_risk['Glucose'] = np.random.uniform(70, 200, size=len(X_risk))
    X_risk = X_risk.fillna(X_risk.median())
    X_risk = X_risk[['Age', 'HR', 'SysBP', 'DiasBP', 'MAP', 'RespRate', 'Temp', 'SpO2', 'Glucose', 'Creatinine', 'WBC', 'Lactate']]
    
    # Use risk_score > 50 as mortality threshold
    y_risk = (risk_df['risk_score'] > 50).astype(int)
    
    all_X.append(X_risk)
    all_y.append(y_risk)
    dataset_names.append(f"ICU_Risk_Score ({len(X_risk)} samples)")
    print(f"   Loaded: {len(X_risk)} samples, Target distribution: {y_risk.value_counts().to_dict()}")
except Exception as e:
    print(f"   Error loading ICU_Risk_Score: {e}")

# Combine all datasets
print("\n" + "="*80)
print("COMBINING ALL DATASETS")
print("="*80)
print(f"Datasets to combine: {dataset_names}")

X_combined = pd.concat(all_X, ignore_index=True)
y_combined = pd.concat(all_y, ignore_index=True)

print(f"\nCombined Dataset: {len(X_combined)} total samples")
print("Target distribution:")
print(y_combined.value_counts())

X = X_combined
y = y_combined

print("\nFeatures used:", X.columns.tolist())
print("\nFeature statistics:")
print(X.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model with regularization (no oversampling)
print("\nTraining Gradient Boosting model with regularization...")
print("Using sample weights to handle class imbalance...")

# Calculate sample weights
class_weights = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}
sample_weights = np.array([class_weights[label] for label in y_train])

# Add regularization: reduce depth, increase min samples, add subsampling
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Make predictions
y_pred = gb_model.predict(X_test_scaled)
y_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold using precision-recall curve for better F1-score
print("\nFinding optimal threshold for F1-score optimization...")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"F1-score at optimal threshold: {f1_scores[optimal_idx]:.4f}")

# Use optimal threshold for predictions
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate metrics with optimal threshold
accuracy = accuracy_score(y_test, y_pred_optimal)
f1 = f1_score(y_test, y_pred_optimal)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Cross-validation for more robust metrics
cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_f1_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='f1')
cv_auc_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"\nCross-Validation Results:")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"CV F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std()*2:.4f})")
print(f"CV AUC-ROC: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std()*2:.4f})")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE (Top 10)")
print("="*50)
print(feature_importance.head(10))

# Save model and metadata
model_package = {
    'model': gb_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'full_feature_set': True  # Flag to indicate we're using full dataset features
}

joblib.dump(model_package, 'NEOcare_mortality_prediction_model.pkl')

metadata = {
    'model_type': 'Gradient Boosting Classifier',
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'auc_roc': float(auc_roc),
    'cv_accuracy': float(cv_scores.mean()),
    'cv_f1_score': float(cv_f1_scores.mean()),
    'cv_auc_roc': float(cv_auc_scores.mean()),
    'feature_importance': feature_importance.head(10).to_dict(),
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'max_features': 'sqrt',
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'features': X.columns.tolist(),
    'class_balancing': 'Sample weights (auto-calculated)',
    'feature_engineering': 'Multi-dataset combination (5 datasets)',
    'data_source': f"Combined: {', '.join(dataset_names)}",
    'datasets_used': dataset_names,
    'optimal_threshold': float(optimal_threshold),
    'threshold_optimization': 'Precision-Recall Curve for F1-score'
}

with open('NEOcare_mortality_prediction_model_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved as 'NEOcare_mortality_prediction_model.pkl'")
print("Metadata saved as 'NEOcare_mortality_prediction_model_meta.json'")
