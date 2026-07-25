import os
import random

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)
os.makedirs("sample_data", exist_ok=True)

N_LEARNERS = 60
STATES = ["Karnataka - IN", "Andhra Pradesh - IN", "Delhi - IN", "Kerala - IN",
          "Haryana - IN", "Madhya Pradesh - IN", "Not Available"]
LEARNER_TYPES = ["Student", "Job Seeker", "Other", "Not Available"]
DELIVERY = ["eLearning", "Learning Plan", "Video", "Guidance"]
SOURCES = ["Adobe Learning Manager", "IBM Learning Patterns", "Moodle",
           "Your Learning Builder - Plans"]
AGES = ["18-25", "26-40", "Not Available"]
ACTIVITIES = [f"Course {chr(65 + i)}" for i in range(20)]

rows = []
for i in range(N_LEARNERS):
    learner_id = 900000 + i
    name = f"Demo Learner {i + 1:02d}"
    ltype = random.choices(LEARNER_TYPES, weights=[5, 2, 1, 3])[0]
    delivery = random.choices(DELIVERY, weights=[6, 3, 1, 1])[0]
    source = random.choices(SOURCES, weights=[5, 4, 1, 2])[0]
    state = random.choice(STATES)
    age = random.choices(AGES, weights=[3, 1, 6])[0]

    engaged = random.random() < 0.4
    n_courses = random.randint(8, 40) if engaged else random.randint(1, 4)

    reg_day = random.randint(1, 28)
    reg_month = random.choice([1, 2, 3])
    reg_date = f"{reg_day:02d}-{reg_month:02d}-2026"

    for _ in range(n_courses):
        days_after = random.randint(0, 25) if engaged else random.randint(0, 2)
        comp = pd.Timestamp(year=2026, month=reg_month, day=reg_day) + pd.Timedelta(days=days_after)
        rows.append({
            "Learner - ID": learner_id,
            "Learner - Name": name,
            "Learner - Type": ltype,
            "Learning activity - ID": f"ACT{random.randint(1000, 9999)}",
            "Learning activity - Title": random.choice(ACTIVITIES),
            "Learning activity - Duration": random.choice([10, 30, 60, 120, 240]),
            "Delivery Type": delivery,
            "Learning Source Name": source,
            "State": state,
            "Age At Registration": age,
            "User Registration Date": reg_date,
            "Completion Date": comp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Learning Last Accessed Date": comp.strftime("%Y-%m-%dT%H:%M:%SZ") if random.random() < 0.3 else "",
            "Source": "SELFREG",
            "Transcript status": "Completed",
            "Transcript Type": "Source Managed",
            "Learning Activity Status": "Active in default YL search",
            "Referrer Source": "Other",
            "College": f"Demo College {random.randint(1, 20)}",
            "Percent complete": random.choice([100, 100, 100, 50, 0]),
            "Learner Enrollment Date": reg_date,
        })

df = pd.DataFrame(rows)
df.to_csv("sample_data/sample_transcript.csv", index=False)
print(f"Wrote {len(df)} rows for {N_LEARNERS} synthetic learners to sample_data/sample_transcript.csv")