
import numpy as np
import pandas as pd

# Load the processed patient data
data = np.load("mimic-iii-clinical-database-demo-1.4/mimic_demo_seq24.npz")

sequences = data["sequences"]
features = data["feature_names"].tolist()

rows = []

# Go through all ICU stays and all 24 time steps
for stay in range(sequences.shape[0]):
    for hour in range(sequences.shape[1]):
        row = {
            "ICU_Stay_Number": stay + 1,
            "Time_Step": hour + 1
        }

        for i, feature in enumerate(features):
            row[feature] = sequences[stay, hour, i]

        rows.append(row)

# Save as CSV
df = pd.DataFrame(rows)
df.to_csv("all_75_icu_stays.csv", index=False)

print("Done bro! CSV created successfully.")
print("Data shape:", sequences.shape)
print("CSV rows:", len(df))
