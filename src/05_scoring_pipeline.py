"""
Step 5 - Scoring Pipeline
"""

import pandas as pd
import numpy as np
import joblib
import os
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ── CONFIG 
# Score the full cleaned dataset (or point to a new month's cleaned files)
CLEANED_FOLDER = 'data/cleaned/*.csv'
MODEL_PATH     = 'models/best_model.pkl'
ENCODERS_PATH  = 'models/encoders.pkl'
FEATURES_PATH  = 'models/feature_names.pkl'
OUTPUT_FOLDER  = 'output/priority_lists/'


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
today    = datetime.today().strftime('%Y_%m_%d')
OUT_FILE = os.path.join(OUTPUT_FOLDER, f'priority_list_{today}.xlsx')

print("="*60)
print("  Step 5 — Credential Priority List Generator")
print("="*60)

# ── LOAD DATA ──
print("\n--- Loading cleaned data ---")
files = sorted(glob.glob(CLEANED_FOLDER))
dfs = [pd.read_csv(f, low_memory=False) for f in files]
trans = pd.concat(dfs, ignore_index=True)
print(f"Rows: {len(trans):,} | Learners: {trans['Learner - ID'].nunique():,}")

# Keep learner names for the output (if present)
if 'Learner - Name' in trans.columns:
    names = trans.drop_duplicates('Learner - ID')[['Learner - ID','Learner - Name']]
else:
    names = None

# ── BUILD SAME FEATURES AS TRAINING ──
print("\n--- Building features ---")
trans['time_on_platform_days'] = pd.to_numeric(
    trans['time_on_platform_days'], errors='coerce')

learner = trans.groupby('Learner - ID').agg(
    total_courses    = ('Learning activity - ID', 'nunique'),
    time_on_platform = ('time_on_platform_days', 'max'),
    learner_type     = ('Learner - Type', 'first'),
    learning_source  = ('Learning Source Name', 'first'),
    delivery_type    = ('Delivery Type', 'first'),
    state            = ('State', 'first'),
    age              = ('Age At Registration', 'first'),
).reset_index()

# Clean categoricals (same as preprocessing)
learner['state'] = learner['state'].str.replace(' - IN', '', regex=False)
for col in ['learner_type','learning_source','delivery_type','state','age']:
    learner[col] = learner[col].replace('Not Available','Unknown').fillna('Unknown')

learner['time_on_platform'] = learner['time_on_platform'].fillna(
    learner['time_on_platform'].median())

print(f"Learners to score: {len(learner):,}")

# ── ENCODE USING SAVED ENCODERS ──
print("\n--- Encoding features ---")
encoders      = joblib.load(ENCODERS_PATH)
feature_names = joblib.load(FEATURES_PATH)

X = learner.copy()
for col, le in encoders.items():
    known = set(le.classes_)
    # Map unseen categories to a known class to avoid errors
    X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
    X[col] = le.transform(X[col].astype(str))

X_score = X[feature_names]

# ── SCORE ──
print("\n--- Scoring learners ---")
model = joblib.load(MODEL_PATH)
proba_earn = model.predict_proba(X_score)[:, 1]   # prob of earning credential

learner['credential_probability'] = (proba_earn * 100).round(1)
# Priority = inverse: low credential probability = high outreach priority
learner['outreach_priority_score'] = (100 - learner['credential_probability']).round(1)

learner['priority_category'] = pd.cut(
    learner['outreach_priority_score'],
    bins=[0, 33, 66, 100],
    labels=['Low Priority','Medium Priority','High Priority'],
    include_lowest=True
)

print("Priority breakdown:")
print(learner['priority_category'].value_counts())

# ── BUILD OUTPUT ──
print("\n--- Building priority list ---")
out = learner.copy()
if names is not None:
    out = out.merge(names, on='Learner - ID', how='left')

cols = {}
if 'Learner - Name' in out.columns:
    cols['Learner - Name'] = 'Learner Name'
cols.update({
    'Learner - ID':             'Learner ID',
    'outreach_priority_score':  'Outreach Priority Score',
    'credential_probability':   'Credential Probability (%)',
    'priority_category':        'Priority',
    'total_courses':            'Total Courses',
    'time_on_platform':         'Days on Platform',
    'learner_type':             'Learner Type',
    'learning_source':          'Learning Source',
    'delivery_type':            'Delivery Type',
    'state':                    'State',
})
out = out.rename(columns=cols)[[v for v in cols.values()]]
out = out.sort_values('Outreach Priority Score', ascending=False).reset_index(drop=True)
out.index = out.index + 1

# ── SAVE EXCEL ──
print("\n--- Saving priority list ---")
with pd.ExcelWriter(OUT_FILE, engine='openpyxl') as writer:
    out.to_excel(writer, sheet_name='Priority List', index=True, index_label='Rank')
    high = out[out['Priority']=='High Priority']
    high.to_excel(writer, sheet_name='High Priority', index=True, index_label='Rank')
    summary = pd.DataFrame({
        'Metric':['Report Date','Total Learners Scored','High Priority',
                  'Medium Priority','Low Priority'],
        'Value':[today, len(out),
                 (out['Priority']=='High Priority').sum(),
                 (out['Priority']=='Medium Priority').sum(),
                 (out['Priority']=='Low Priority').sum()]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"  Saved: {OUT_FILE}")

print(f"\n{'='*60}")
print(f"  Scoring Complete")
print(f"{'='*60}")
print(f"  Total scored:    {len(out):,}")
print(f"  High Priority:   {(out['Priority']=='High Priority').sum():,}")
print(f"  Medium Priority: {(out['Priority']=='Medium Priority').sum():,}")
print(f"  Low Priority:    {(out['Priority']=='Low Priority').sum():,}")
print(f"\n  High Priority = least likely to earn credential = reach out first")
