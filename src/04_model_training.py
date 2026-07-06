"""
Step 4 - Model Training & Selection
=====================================
- Loads learner-level data
- Encodes features, stratified train/test split
- Applies SMOTE for class imbalance
- Trains & compares Logistic Regression, Random Forest, XGBoost
- Evaluates with proper metrics + cross-validation
- Saves charts and best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Try to import XGBoost; fall back gracefully if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Note: XGBoost not installed. Run: pip install xgboost")

# ── CONFIG ──────────────────────────────────────────────────
INPUT_FILE    = 'data/combined/learner_level_data.csv'
MODEL_FOLDER  = 'models/'
OUTPUT_FOLDER = 'output/model_results/'
# ────────────────────────────────────────────────────────────

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

C = {'navy':'#1A2744','teal':'#00A896','orange':'#F97316',
     'red':'#E11D48','green':'#059669','light':'#EEF2F7'}

print("="*60)
print("  Step 4 — Model Training & Selection")
print("="*60)

# ── LOAD ──
df = pd.read_csv(INPUT_FILE)
print(f"\nLearners: {len(df):,}")
print(f"Earned: {(df['earned_credential']==1).sum():,} ({(df['earned_credential']==1).mean()*100:.1f}%)")

# ── FEATURES / TARGET ──
DROP = ['Learner - ID', 'earned_credential']
FEATURES = [c for c in df.columns if c not in DROP]
X = df[FEATURES].copy()
y = df['earned_credential'].copy()

cat_cols = ['learner_type','learning_source','delivery_type','state','age']

# ── ENCODE ──
print("\n--- Encoding categorical features ---")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categories")
joblib.dump(encoders, os.path.join(MODEL_FOLDER, 'encoders.pkl'))

# ── SPLIT ──
print("\n--- Train/test split (stratified 80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── SMOTE ──
print("\n--- Applying SMOTE to training set ---")
print(f"  Before: earned {y_train.sum():,} | not {(y_train==0).sum():,}")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"  After:  earned {y_train_sm.sum():,} | not {(y_train_sm==0).sum():,}")

# ── MODELS ──
print("\n--- Training models ---")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}
if HAS_XGB:
    models['XGBoost'] = XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)

results = {}
for name, model in models.items():
    print(f"\n  {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    r = results[name]
    print(f"    Accuracy {r['accuracy']:.3f} | Precision {r['precision']:.3f} | "
          f"Recall {r['recall']:.3f} | F1 {r['f1']:.3f} | AUC {r['roc_auc']:.3f}")

# ── COMPARISON TABLE ──
print("\n--- Model comparison ---")
comp = pd.DataFrame({n:{'Accuracy':r['accuracy'],'Precision':r['precision'],
    'Recall':r['recall'],'F1':r['f1'],'ROC_AUC':r['roc_auc']}
    for n,r in results.items()}).T
print(comp.round(3).to_string())
comp.to_csv(os.path.join(OUTPUT_FOLDER,'model_comparison.csv'))

# ── BEST MODEL (by F1) ──
best_name = max(results, key=lambda x: results[x]['f1'])
best = results[best_name]
print(f"\nBest model (by F1): {best_name}")

# ── CROSS VALIDATION on best ──
print("\n--- 5-fold cross-validation (best model) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best['model'], X_train_sm, y_train_sm, cv=cv, scoring='f1')
print(f"  CV F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean CV F1:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── CHARTS ──
print("\n--- Saving charts ---")

# Comparison bars
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', color=C['navy'])
metrics = ['accuracy','precision','recall','f1','roc_auc']
labels = ['Accuracy','Precision','Recall','F1','ROC AUC']
colors = [C['navy'],C['teal'],C['orange'],C['red'],C['green']]
for ax, m, lab, col in zip(axes, metrics, labels, colors):
    ax.set_facecolor(C['light'])
    names = list(results.keys())
    vals = [results[n][m] for n in names]
    bars = ax.bar(range(len(names)), vals, color=col, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(lab, fontsize=12, fontweight='bold', color=C['navy'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0,1.1)
    ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER,'model_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results),5))
if len(results)==1: axes=[axes]
fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold', color=C['navy'])
for ax,(name,r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['No Cred','Earned'], yticklabels=['No Cred','Earned'])
    ax.set_title(name, fontsize=11, fontweight='bold', color=C['navy'])
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER,'confusion_matrices.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8,6))
ax.set_facecolor(C['light'])
roc_colors=[C['navy'],C['teal'],C['orange']]
for (name,r),col in zip(results.items(), roc_colors):
    fpr,tpr,_ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr,tpr,color=col,linewidth=2.5,label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0,1],[0,1],'k--',alpha=0.5)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold', color=C['navy'])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(); ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER,'roc_curves.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  roc_curves.png")

# Feature importance (best tree model)
imp_model = best['model']
if hasattr(imp_model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature':X.columns,'Importance':imp_model.feature_importances_}).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_facecolor(C['light'])
    bars = ax.barh(fi['Feature'], fi['Importance'], color=C['teal'], edgecolor='white', height=0.6)
    for bar,v in zip(bars, fi['Importance']):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{v:.3f}',
                va='center', fontsize=9, fontweight='bold')
    ax.set_title(f'Feature Importance — {best_name}', fontsize=14, fontweight='bold', color=C['navy'])
    ax.set_xlabel('Importance'); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER,'feature_importance.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  feature_importance.png")

# ── SAVE BEST MODEL ──
joblib.dump(best['model'], os.path.join(MODEL_FOLDER,'best_model.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_FOLDER,'feature_names.pkl'))
print(f"\n  Best model saved: models/best_model.pkl")

print(f"\n{'='*60}")
print(f"  Training Complete")
print(f"{'='*60}")
print(f"  Best Model: {best_name}")
print(f"  F1: {best['f1']:.3f} | Recall: {best['recall']:.3f} | AUC: {best['roc_auc']:.3f}")
print(f"  CV F1: {cv_scores.mean():.3f}")
print(f"\n  Next: 05_scoring_pipeline.py")