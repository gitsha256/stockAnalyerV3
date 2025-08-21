import pandas as pd
import glob
import os

# Find the latest file matching the pattern
files = glob.glob("*-*-25snapshot.csv")
if not files:
    print("No snapshot file found matching pattern *-*-25snapshot.csv")
    exit(1)
latest_file = max(files, key=os.path.getmtime)
print(f"Using file: {latest_file}")

df = pd.read_csv(latest_file)
threshold = float(input("Enter the percent threshold (e.g., 5 for 5%): ")) / 100

# Drop rows with missing SMA or symbol values
sma_cols = ['s020', 's050', 's100', 's200', 'symb']
df = df.dropna(subset=sma_cols)

def all_near(row):
    sma200 = row['s200']
    return (
        abs(row['s020'] - sma200) / sma200 <= threshold and
        abs(row['s050'] - sma200) / sma200 <= threshold and
        abs(row['s100'] - sma200) / sma200 <= threshold
    )

matching = df[df.apply(all_near, axis=1)]

# Save unique symbols to file
matching[['symb']].drop_duplicates().to_csv("sma.csv", index=False)
print(matching[['symb']].drop_duplicates())

print("Rows after dropna:", len(df))
print("Rows matching SMA confluence:", len(matching))
print("Threshold:", threshold)