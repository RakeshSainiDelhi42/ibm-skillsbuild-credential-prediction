"""
Step 1 - Clean & Prepare the 14 Transcript Files
=================================================
For each of the 14 raw files:
- Adds 2 new columns:
    * last_activity_date      = Learning Last Accessed Date if present,
                                else Completion Date
    * time_on_platform_days   = last_activity_date - User Registration Date
- Drops columns that are definitely useless (constant / empty / redundant)
- Keeps anything uncertain (we'll analyze those next)
- Saves 14 cleaned files to data/cleaned/

Raw files are never modified.
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────
RAW_FOLDER     = 'data/raw/*.csv'
CLEANED_FOLDER = 'data/cleaned/'
# ────────────────────────────────────────────────────────────

os.makedirs(CLEANED_FOLDER, exist_ok=True)

print("="*60)
print("  Step 1 — Clean & Prepare 14 Files")
print("="*60)

# Columns to DROP — definitely useless (verified in analysis)
DROP_COLS = [
    'Cancellation Type',          # 0% filled / single value
    'Vendor',                     # 415,140 'Not Available', 2 others
    'Learner Status',             # 415,128 'Active' — near constant
    'Delivery Category',          # near constant 'Self-paced'
    'Geography',                  # 414,882 'ISA' — near constant
    'Completed date',             # exact duplicate of Completion Date
    'Transcript Date',            # system field, not useful
    'User Registration Date Range',  # bucketed duplicate of reg date
    'Percent Of Expected Duration Complete',  # 6.2% filled
    'Expected Learning Duration',  # 6.2% filled
    'Learner Cancellation Date',  # 2.5% filled
    'Other Referrer Source',      # admin/sparse text
    'Other Learner Type',         # sparse text
    'Other College',              # sparse text
    'Catalog Groupspace Name',    # not useful for learner model
    'Learner - Name',             # PII — not a feature (kept only if needed for output later)
]

# ─────────────────────────────────────────────────────────────
# PROCESS EACH FILE
# ─────────────────────────────────────────────────────────────

all_files = sorted(glob.glob(RAW_FOLDER))
print(f"\nFound {len(all_files)} files\n")

for f in all_files:
    fname = os.path.basename(f)
    df = pd.read_csv(f, low_memory=False)
    original_cols = df.shape[1]

    # --- Build last_activity_date ---
    last_access = pd.to_datetime(
        df['Learning Last Accessed Date'], format='ISO8601',
        errors='coerce', utc=True)
    completion = pd.to_datetime(
        df['Completion Date'], format='ISO8601',
        errors='coerce', utc=True)
    df['last_activity_date'] = last_access.fillna(completion)

    # --- Build time_on_platform_days ---
    registration = pd.to_datetime(
        df['User Registration Date'], format='%d-%m-%Y',
        errors='coerce', utc=True)
    df['time_on_platform_days'] = (
        df['last_activity_date'] - registration).dt.days

    # Guard: negative values (data inconsistency) set to NaN
    df.loc[df['time_on_platform_days'] < 0, 'time_on_platform_days'] = np.nan

    # --- Drop useless columns ---
    drop_present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_present)

    # --- Save cleaned file ---
    out_path = os.path.join(CLEANED_FOLDER, fname)
    df.to_csv(out_path, index=False)

    # Report
    has_activity = df['last_activity_date'].notna().mean()*100
    has_time     = df['time_on_platform_days'].notna().mean()*100
    print(f"  ✓  {fname[:42]:42s} | {original_cols}->{df.shape[1]} cols | "
          f"activity {has_activity:4.0f}% | time {has_time:4.0f}%")

print(f"\n{'='*60}")
print(f"  Cleaning Complete")
print(f"{'='*60}")
print(f"  14 cleaned files saved to: {CLEANED_FOLDER}")
print(f"  Added columns: last_activity_date, time_on_platform_days")
print(f"  Dropped {len(DROP_COLS)} useless columns")
print(f"\n  Next step: 02_data_analysis.py on the cleaned files")