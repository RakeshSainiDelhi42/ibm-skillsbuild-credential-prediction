"""
Step 4b - Model Diagnostics (Honesty & Robustness Checks)
==========================================================
1. Full classification report (per-class precision/recall)
2. Confusion matrix numbers
3. Drop-delivery_type test — is the model one-feature-dependent?
4. Each-feature-alone test — how much does each single feature predict?
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

INPUT_FILE = 'data/combined/learner_level_data.csv'

print("="*60)
print("  Step 4b — Model Diagnostics")
print("="*60)

df = pd.read_csv(INPUT_FILE)
DROP = ['Learner - ID','earned_credential']
FEATURES = [c for c in df.columns if c not in DROP]
cat_cols = ['learner_type','learning_source','delivery_type','state','age']

# Encode
X = df[FEATURES].copy()
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
y = df['earned_credential']

def make_model():
    if HAS_XGB:
        return XGBClassifier(n_estimators=100, random_state=42,
                             eval_metric='logloss', use_label_encoder=False)
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def train_eval(Xf, label):
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2,
                                          random_state=42, stratify=y)
    Xtr_sm, ytr_sm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    m = make_model()
    m.fit(Xtr_sm, ytr_sm)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)[:,1]
    return f1_score(yte, pred), roc_auc_score(yte, proba), yte, pred

# ── CHECK 1 — Full classification report on all features ──
print("\n" + "="*60)
print("  CHECK 1 — Classification Report (all features)")
print("="*60)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xtr_sm, ytr_sm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
model = make_model()
model.fit(Xtr_sm, ytr_sm)
pred = model.predict(Xte)
print(classification_report(yte, pred, target_names=['No Credential','Earned Credential']))

# ── CHECK 2 — Confusion matrix numbers ──
print("="*60)
print("  CHECK 2 — Confusion Matrix")
print("="*60)
cm = confusion_matrix(yte, pred)
print(f"\n                    Predicted No   Predicted Earned")
print(f"  Actual No Cred:      {cm[0,0]:>6,}         {cm[0,1]:>6,}")
print(f"  Actual Earned:       {cm[1,0]:>6,}         {cm[1,1]:>6,}")
print(f"\n  Earned learners correctly caught: {cm[1,1]:,} / {cm[1,0]+cm[1,1]:,}")
print(f"  Earned learners missed:           {cm[1,0]:,}")

# ── CHECK 3 — Drop delivery_type ──
print("\n" + "="*60)
print("  CHECK 3 — Performance WITHOUT delivery_type")
print("="*60)
full_f1, full_auc, _, _ = train_eval(X, "full")
no_dt_f1, no_dt_auc, _, _ = train_eval(X.drop(columns=['delivery_type']), "no_dt")
print(f"\n  WITH delivery_type:    F1 {full_f1:.3f} | AUC {full_auc:.3f}")
print(f"  WITHOUT delivery_type: F1 {no_dt_f1:.3f} | AUC {no_dt_auc:.3f}")
print(f"  Drop in F1: {full_f1 - no_dt_f1:.3f}")
if full_f1 - no_dt_f1 > 0.15:
    print("  -> Model leans HEAVILY on delivery_type")
else:
    print("  -> Other features carry real signal too (healthy)")

# ── CHECK 4 — Each feature ALONE ──
print("\n" + "="*60)
print("  CHECK 4 — Each Feature Alone (F1 score)")
print("="*60)
print(f"\n{'Feature':20s} {'F1 alone':>10s} {'AUC alone':>10s}")
print("-"*42)
for feat in FEATURES:
    f1_a, auc_a, _, _ = train_eval(X[[feat]], feat)
    print(f"{feat:20s} {f1_a:>10.3f} {auc_a:>10.3f}")

print("\n" + "="*60)
print("  DIAGNOSTICS COMPLETE — share output")
print("="*60)