"""
Step 2 - Data Analysis on Cleaned Files
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ── CONFIG 
CLEANED_FOLDER = 'data/cleaned/*.csv'
CRED_FILE      = 'data/credentials/V3SkillsReportCredentialsDetail (4).csv'


print("="*70)
print("  DATA ANALYSIS ON CLEANED FILES")
print("="*70)

# ── LOAD 
files = sorted(glob.glob(CLEANED_FOLDER))
dfs = [pd.read_csv(f, low_memory=False) for f in files]
trans = pd.concat(dfs, ignore_index=True)
cred = pd.read_csv(CRED_FILE, low_memory=False)
cred_learners = set(cred['Learner - ID'])

print(f"\nLoaded {len(files)} cleaned files")
print(f"Transcript: {len(trans):,} rows | {trans['Learner - ID'].nunique():,} learners")
print(f"Columns: {trans.shape[1]}")

# ── TARGET 
trans['earned'] = trans['Learner - ID'].isin(cred_learners).astype(int)

#  FILL RATES OF ALL REMAINING COLUMNS 
print("\n" + "="*70)
print("  Column Fill Rates (cleaned files)")
print("="*70)
print(f"{'Column':40s} {'Fill%':>7s} {'Unique':>8s}")
print("-"*60)
for c in trans.columns:
    if c in ['snapshot_file','earned']: continue
    print(f"{c:40s} {trans[c].notna().mean()*100:6.1f}% {trans[c].nunique():>8,}")

#  BUILD LEARNER-LEVEL SUMMARY 
print("\n" + "="*70)
print("  Building Learner-Level Features")
print("="*70)

trans['time_on_platform_days'] = pd.to_numeric(trans['time_on_platform_days'], errors='coerce')
trans['Percent complete'] = pd.to_numeric(trans['Percent complete'], errors='coerce')

learner = trans.groupby('Learner - ID').agg(
    total_courses   = ('Learning activity - ID','nunique'),
    time_on_platform= ('time_on_platform_days','max'),
    learner_type    = ('Learner - Type','first'),
    state           = ('State','first'),
    delivery_type   = ('Delivery Type','first'),
    source          = ('Source','first'),
    learning_source = ('Learning Source Name','first'),
    age             = ('Age At Registration','first'),
    transcript_type = ('Transcript Type','first'),
    activity_status = ('Learning Activity Status','first'),
    avg_pct         = ('Percent complete','mean'),
    earned          = ('earned','first'),
).reset_index()

print(f"Learners: {len(learner):,}")
print(f"Earned credential: {(learner['earned']==1).sum():,} ({(learner['earned']==1).mean()*100:.1f}%)")

# ═══ NUMERIC FEATURE SEPARATION ═══
print("\n" + "="*70)
print("  Numeric Features — Separation (earned vs not)")
print("="*70)
for col in ['total_courses','time_on_platform','avg_pct']:
    v = learner[learner[col].notna()]
    g = v.groupby('earned')[col].agg(['mean','median'])
    print(f"\n{col}:")
    print(f"  Earned:    mean {g.loc[1,'mean']:8.1f} | median {g.loc[1,'median']:8.1f}")
    print(f"  Not earned:mean {g.loc[0,'mean']:8.1f} | median {g.loc[0,'median']:8.1f}")

# CATEGORICAL FEATURE SEPARATION 
print("\n" + "="*70)
print("  Categorical Features — Credential Rate by Category")
print("="*70)
for cat in ['learner_type','delivery_type','source','learning_source',
            'age','transcript_type','activity_status','state']:
    print(f"\n-- {cat} --")
    r = learner.groupby(cat)['earned'].agg(['mean','count'])
    r['mean'] = (r['mean']*100).round(1)
    r = r[r['count']>=50].sort_values('mean',ascending=False)
    r.columns = ['cred_rate_%','count']
    print(r.head(8).to_string())

# MISSING VALUES AT LEARNER LEVEL
print("\n" + "="*70)
print("  Missing Values (learner level)")
print("="*70)
miss = (learner.isnull().sum()/len(learner)*100).round(1)
print(miss[miss>0].sort_values(ascending=False).to_string())

print("\n" + "="*70)
print("  ANALYSIS COMPLETE — share all output")
print("="*70)
