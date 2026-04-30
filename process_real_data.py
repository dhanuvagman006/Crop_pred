import pandas as pd

# Load the full dataset
df = pd.read_csv('real_india_crop_production.csv')

# Clean column names (strip whitespace)
df.columns = [c.strip() for c in df.columns]

# Filter for Dakshina Kannada (case insensitive and handle potential name variations)
dk_df = df[
    (df['State_Name'].str.contains('Karnataka', case=False, na=False)) &
    (df['District_Name'].str.contains('DAKSHIN KANNAD', case=False, na=False))
].copy()

# Remove duplicates
initial_len = len(dk_df)
dk_df.drop_duplicates(inplace=True)
final_len = len(dk_df)

print(f"Initial records for DK: {initial_len}")
print(f"Records after removing duplicates: {final_len}")

# Calculate Yield (Production / Area)
dk_df['Yield'] = dk_df['Production'] / dk_df['Area']

# Drop rows where yield is NaN (production was missing)
dk_df.dropna(subset=['Yield'], inplace=True)

# Save the cleaned dataset
dk_df.to_csv('DK_Real_CropYield.csv', index=False)

print(f"Saved real dataset to DK_Real_CropYield.csv with {len(dk_df)} records.")
print("\nUnique Crops in DK:")
print(dk_df['Crop'].unique())
