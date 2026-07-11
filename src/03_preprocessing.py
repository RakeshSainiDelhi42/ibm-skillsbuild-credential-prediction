"""
Step 3 - Preprocessing & Feature Engineering

"""

import pandas as pd
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings('ignore')

# ── CONFIG 
CLEANED_FOLDER = 'data/cleaned/*.csv'
CRED_FILE      = 'data/credentials/V3SkillsReportCredentialsDetail (4).csv'
OUTPUT_FILE    = 'data/combined/learner_level_data.csv'


os.makedirs('data/combined/', exist_ok=True)

print("="*60)
print("  Step 3 — Preprocessing & Feature Engineering")
print("="*60)

# ── LOAD 
print("\n--- Loading cleaned files + credential file ---")
files = sorted(glob.glob(CLEANED_FOLDER))
dfs = [pd.read_csv(f, low_memory=False) for f in files]
trans = pd.concat(dfs, ignore_index=True)
cred = pd.read_csv(CRED_FILE, low_memory=False)
cred_learners = set(cred['Learner - ID'])
print(f"Transcript rows: {len(trans):,} | Learners: {trans['Learner - ID'].nunique():,}")

# ── NUMERIC PREP 
trans['time_on_platform_days'] = pd.to_numeric(
    trans['time_on_platform_days'], errors='coerce')

# ── AGGREGATE TO LEARNER LEVEL 
print("\n--- Aggregating to learner level ---")
learner = trans.groupby('Learner - ID').agg(
    total_courses    = ('Learning activity - ID', 'nunique'),
    time_on_platform = ('time_on_platform_days', 'max'),
    learner_type     = ('Learner - Type', 'first'),
    learning_source  = ('Learning Source Name', 'first'),
    delivery_type    = ('Delivery Type', 'first'),
    state            = ('State', 'first'),
    age              = ('Age At Registration', 'first'),
).reset_index()

print(f"Learners after aggregation: {len(learner):,}")

# ── TARGET ──
print("\n--- Creating target ---")
learner['earned_credential'] = learner['Learner - ID'].isin(
    cred_learners).astype(int)
print(learner['earned_credential'].value_counts())
print(f"Earned: {(learner['earned_credential']==1).mean()*100:.1f}%")

# ── CLEAN CATEGORICALS ──
print("\n--- Cleaning categorical features ---")
# Clean state suffix
learner['state'] = learner['state'].str.replace(' - IN', '', regex=False)

# Replace Not Available with Unknown for categoricals
for col in ['learner_type', 'learning_source', 'delivery_type', 'state', 'age']:
    learner[col] = learner[col].replace('Not Available', 'Unknown')
    learner[col] = learner[col].fillna('Unknown')

# ── HANDLE MISSING NUMERIC 
print("\n--- Handling missing numeric values ---")
# time_on_platform: fill missing with median
med = learner['time_on_platform'].median()
missing_time = learner['time_on_platform'].isna().sum()
learner['time_on_platform'] = learner['time_on_platform'].fillna(med)
print(f"  time_on_platform: filled {missing_time:,} missing with median ({med:.0f})")

# ── MISSING CHECK 
print("\n--- Final missing values ---")
miss = (learner.isnull().sum()/len(learner)*100).round(1)
miss = miss[miss>0]
if len(miss)==0:
    print("  No missing values remaining")
else:
    print(miss.to_string())

# ── SAVE 
learner.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*60}")
print(f"  Preprocessing Complete")
print(f"{'='*60}")
print(f"  Learners:          {len(learner):,}")
print(f"  Earned credential: {(learner['earned_credential']==1).sum():,}")
print(f"  No credential:     {(learner['earned_credential']==0).sum():,}")
features = [c for c in learner.columns if c not in ['Learner - ID','earned_credential']]
print(f"  Features ({len(features)}): {features}")
print(f"  Saved to: {OUTPUT_FILE}")
print(f"\n  Next step: 04_model_training.py")
