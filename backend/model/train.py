"""
EventIQ v2 — Training Pipeline
Model: Gradient Boosting Classifier
Dataset: event_dataset_3000.csv  (3000 rows, 15 raw features + 2 new: organizer_score, social_buzz)
Target: attended (binary 0/1)
"""
import warnings
warnings.filterwarnings('ignore')
import os, pandas as pd, numpy as np, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '..', '..', 'data', 'event_dataset_3000.csv')
MODEL_OUT  = os.path.join(BASE_DIR, 'model.pkl')
SCALER_OUT = os.path.join(BASE_DIR, 'scaler.pkl')
ENC_OUT    = os.path.join(BASE_DIR, 'encoders.pkl')
META_OUT   = os.path.join(BASE_DIR, 'meta.pkl')

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}  |  Attendance rate: {df['attended'].mean():.2%}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
num_cols = ['age','income','event_rating','distance_km','previous_events',
            'organizer_score','social_buzz']
cat_cols = ['gender','location','event_type']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ── Date features ─────────────────────────────────────────────────────────────
df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
df['event_date']        = pd.to_datetime(df['event_date'],        errors='coerce')
df.dropna(subset=['registration_date','event_date'], inplace=True)
df['days_to_event']     = (df['event_date'] - df['registration_date']).dt.days
df['event_month']       = df['event_date'].dt.month
df['reg_month']         = df['registration_date'].dt.month
df['event_dow']         = df['event_date'].dt.dayofweek
df['is_weekend']        = (df['event_dow'] >= 5).astype(int)
df['event_quarter']     = df['event_date'].dt.quarter
df.drop(['registration_date','event_date'], axis=1, inplace=True)

# ── Interaction features ──────────────────────────────────────────────────────
df['rating_x_org']   = df['event_rating'] * df['organizer_score']
df['buzz_x_rating']  = df['social_buzz']  * df['event_rating'] / 100
df['prev_x_org']     = df['previous_events'] * df['organizer_score']
df['income_per_km']  = df['income'] / (df['distance_km'] + 1)
df['rating_x_prev']  = df['event_rating'] * df['previous_events']
df['dist_income']    = df['distance_km']  / (df['income'] / 10000 + 1)
df['loyalty_score']  = (df['previous_events'] * df['event_rating'] *
                        df['organizer_score'] / 25)

# ── Encode ────────────────────────────────────────────────────────────────────
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── Outlier removal (IQR on original num cols only) ───────────────────────────
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# ── Scale ─────────────────────────────────────────────────────────────────────
feat_cols = [c for c in df.columns if c not in ['user_id','event_id','attended']]
X = df[feat_cols]
y = df['attended']

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols)
print(f"  After cleaning: {X_sc.shape}  |  Features: {len(feat_cols)}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators     = 300,
    learning_rate    = 0.05,
    max_depth        = 5,
    subsample        = 0.85,
    min_samples_split= 5,
    random_state     = 42,
)

Xtr, Xte, ytr, yte = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(Xtr, ytr)

# ── Evaluate ──────────────────────────────────────────────────────────────────
yp   = model.predict(Xte)
ypr  = model.predict_proba(Xte)[:,1]
acc  = accuracy_score(yte, yp)
auc  = roc_auc_score(yte, ypr)
f1   = f1_score(yte, yp)
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X_sc, y, cv=cv, scoring='accuracy').mean()

print(f"\n  Test Accuracy : {acc:.4f}")
print(f"  AUC-ROC       : {auc:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  5-Fold CV Acc : {cv_acc:.4f}")
print(f"\n{classification_report(yte, yp)}")

imp = sorted(zip(feat_cols, model.feature_importances_),
             key=lambda x: x[1], reverse=True)
print("Feature importances:")
for fn, fv in imp:
    print(f"  {fn:<25} {fv:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
cm = confusion_matrix(yte, yp).tolist()
joblib.dump(model,    MODEL_OUT)
joblib.dump(scaler,   SCALER_OUT)
joblib.dump(encoders, ENC_OUT)
joblib.dump({
    'feature_cols':    feat_cols,
    'cat_cols':        cat_cols,
    'num_cols':        num_cols,
    'metrics':         {'accuracy': round(acc,4), 'auc': round(auc,4),
                        'f1': round(f1,4), 'cv_acc': round(cv_acc,4)},
    'importances':     dict(imp),
    'confusion_matrix': cm,
    'class_report':    classification_report(yte, yp, output_dict=True),
    'model_name':      'Gradient Boosting Classifier',
    'dataset_size':    len(df),
    'n_features':      len(feat_cols),
}, META_OUT)
print("\nArtifacts saved: model.pkl  scaler.pkl  encoders.pkl  meta.pkl")
print("Training complete.")
