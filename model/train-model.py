# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import top_k_accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import catboost as cb
import optuna
import joblib

# Load dataset
df = pd.read_csv('combined_cutoff_marks_2019_2024.csv')

# Create composite key
df['university_course'] = df['university'] + '|' + df['course_name']

# Feature engineering
df['zscore_district'] = df['z-score'] * df['district'].astype('category').cat.codes
poly = PolynomialFeatures(degree=2, include_bias=False)
zscore_poly = poly.fit_transform(df[['z-score']])
df['zscore_squared'] = zscore_poly[:, 1]
df['zscore_trend'] = df.groupby(['university_course', 'district'])['z-score'].transform(lambda x: x.diff().fillna(0))
df['stream_district'] = df['stream'].astype(str) + '|' + df['district'].astype(str)

# Encode categoricals
label_encoders = {}
for col in ['stream', 'district', 'university_course', 'stream_district']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numeric columns
scaler = RobustScaler()
numerical_cols = ['z-score', 'zscore_district', 'zscore_squared', 'zscore_trend']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Features/Target
X = df[['z-score', 'stream', 'district', 'year', 'zscore_district', 'zscore_squared', 'zscore_trend', 'stream_district']]
y = df['university_course']

# Balance classes
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- OPTUNA TUNING ---

def optimize_model(trial, model_type):
    if model_type == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=len(label_encoders['university_course'].classes_), random_state=42)
    elif model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        model = RandomForestClassifier(**params, random_state=42)
    elif model_type == 'cb':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
        }
        model = cb.CatBoostClassifier(**params, verbose=0, random_state=42)
    else:
        return 0
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

# Tune models
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda trial: optimize_model(trial, 'xgb'), n_trials=10)
xgb_model = xgb.XGBClassifier(**study_xgb.best_params, objective='multi:softprob', num_class=len(label_encoders['university_course'].classes_), random_state=42)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(lambda trial: optimize_model(trial, 'rf'), n_trials=10)
rf_model = RandomForestClassifier(**study_rf.best_params, random_state=42)

study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(lambda trial: optimize_model(trial, 'cb'), n_trials=10)
cb_model = cb.CatBoostClassifier(**study_cb.best_params, verbose=0, random_state=42)

# STACKING
stage1_model = StackingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model), ('cb', cb_model)],
    final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42)
)
stage1_model.fit(X_train, y_train)

# --- STAGE 2 ---
stage1_probs_train = stage1_model.predict_proba(X_train)
X_train_stage2 = []
y_train_stage2 = []

for i in range(len(X_train)):
    probas = stage1_model.predict_proba(X_train.iloc[i:i+1])[0]
    top10 = np.argsort(probas)[-10:]
    for candidate in top10:
        X_train_stage2.append(np.append(X_train.iloc[i].values, candidate))
        y_train_stage2.append(1 if y_train.iloc[i] == candidate else 0)

X_train_stage2 = np.array(X_train_stage2)
y_train_stage2 = np.array(y_train_stage2)

# Optimize stage 2 model
def stage2_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2)
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    return cross_val_score(model, X_train_stage2, y_train_stage2, cv=3, scoring='accuracy').mean()

study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(stage2_objective, n_trials=10)
stage2_model = GradientBoostingClassifier(**study_gb.best_params, random_state=42)
stage2_model.fit(X_train_stage2, y_train_stage2)

# Save everything
joblib.dump(stage1_model, 'stage1_model.pkl')
joblib.dump(stage2_model, 'stage2_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')
