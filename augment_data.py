import argparse
import pandas as pd
import numpy as np

# CLI: allow disabling synthesis/expansion for reproducible "real-only" datasets
parser = argparse.ArgumentParser(description='Augment DK real crop yield data')
parser.add_argument('--no-synth', action='store_true', help='Do not synthesize missing crop records')
parser.add_argument('--no-expand', action='store_true', help='Do not perform massive synthetic expansion')
args = parser.parse_args()

# Load the real APY data
df = pd.read_csv('DK_Real_CropYield.csv')

# Clean columns
df['Crop'] = df['Crop'].str.strip()
df['Season'] = df['Season'].str.strip()

# Handle Rice/Paddy duplication (they are the same in APY context)
df['Crop'] = df['Crop'].replace('Paddy', 'Rice')

# 1. Remove obvious duplicates
df = df.drop_duplicates()

# 2. ENSURE 8 CROPS (Synthesize if missing or sparse)
TARGET_CROPS = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
years = sorted(df['Crop_Year'].unique())

if not args.no_synth:
    for crop in TARGET_CROPS:
        current_count = len(df[df['Crop'] == crop])
        if current_count < 15:  # If sparse or missing
            print(f"Synthesizing research data for {crop}...")
            new_rows = []
            # Base yields for DK region
            base_yields = {'Cocoa': 600, 'Cashewnut': 500, 'Mango': 5000}
            base_y = base_yields.get(crop, 1000)

            for year in years:
                area = 1500 + np.random.normal(0, 200)
                # Heuristic yield with some noise
                yield_val = base_y + np.random.normal(0, base_y * 0.1)
                prod = (yield_val * area) / 1000

                new_rows.append({
                    'State_Name': 'Karnataka',
                    'District_Name': 'DAKSHIN KANNAD',
                    'Crop_Year': year,
                    'Season': 'Whole Year',
                    'Crop': crop,
                    'Area': area,
                    'Production': prod,
                    'Yield': yield_val,
                    'Is_Synthesized': True
                })
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
else:
    # mark existing rows as real
    df['Is_Synthesized'] = False

# 3. AUGMENT WITH SEMI-REAL WEATHER (Dakshina Kannada specific)
def get_weather(year):
    np.random.seed(int(year) % 2**32)
    rainfall = 3500 + (int(year) % 10) * 50 + np.random.normal(0, 200)
    max_temp = 32 + np.random.normal(0, 1)
    min_temp = 22 + np.random.normal(0, 1)
    humidity = 80 + np.random.normal(0, 5)
    sunshine = 6 + np.random.normal(0, 0.5)
    wind = 12 + np.random.normal(0, 2)
    return rainfall, max_temp, min_temp, humidity, sunshine, wind

def get_soil(crop):
    np.random.seed(abs(hash(crop)) % 2**32)
    ph = 5.5 + np.random.normal(0, 0.3)
    nitrogen = 180 + np.random.normal(0, 20)
    phosphorus = 40 + np.random.normal(0, 5)
    potassium = 150 + np.random.normal(0, 15)
    carbon = 1.2 + np.random.normal(0, 0.2)
    moisture = 35 + np.random.normal(0, 5)
    ec = 0.3 + np.random.normal(0, 0.05)
    return ph, nitrogen, phosphorus, potassium, carbon, moisture, ec

# Filter to keep only the 8 target crops
df = df[df['Crop'].isin(TARGET_CROPS)].reset_index(drop=True)

# Apply augmentations
weather_data = [get_weather(y) for y in df['Crop_Year']]
soil_data = [get_soil(c) for c in df['Crop']]

df['Rainfall'] = [w[0] for w in weather_data]
df['Max Temp'] = [w[1] for w in weather_data]
df['Min Temp'] = [w[2] for w in weather_data]
df['Humidity'] = [w[3] for w in weather_data]
df['Sunshine'] = [w[4] for w in weather_data]
df['Wind Speed'] = [w[5] for w in weather_data]

df['Soil pH'] = [s[0] for s in soil_data]
df['Nitrogen'] = [s[1] for s in soil_data]
df['Phosphorus'] = [s[2] for s in soil_data]
df['Potassium'] = [s[3] for s in soil_data]
df['Organic Carbon'] = [s[4] for s in soil_data]
df['Soil Moisture'] = [s[5] for s in soil_data]
df['EC'] = [s[6] for s in soil_data]

# Final Cleanup: Remove any original duplicates that might have been overshadowed
df = df.drop_duplicates(subset=['Crop', 'Crop_Year', 'Season'])

# 4. MASSIVE SYNTHETIC EXPANSION (To help Deep Learning models generalize)
print(f"Original data size: {len(df)}")
EXPANDED_SAMPLES_PER_CROP = 2000
expanded_df = []

if not args.no_expand:
    for crop in TARGET_CROPS:
        crop_df = df[df['Crop'] == crop].copy()
        if len(crop_df) == 0:
            continue

        # Keep original
        expanded_df.append(crop_df)

        # Generate variations
        num_to_gen = EXPANDED_SAMPLES_PER_CROP - len(crop_df)
        if num_to_gen > 0:
            for _ in range(num_to_gen):
                # Pick a random base row
                base_row = crop_df.sample(1).iloc[0].copy()

                # Add a slight yearly trend (0.5% growth per year relative to mean year)
                mean_year = crop_df['Crop_Year'].mean()
                year_diff = base_row['Crop_Year'] - mean_year
                trend_factor = 1.0 + (year_diff * 0.005)

                # Add jitter to numerical features (1% to 5% noise)
                num_cols = [
                    'Area', 'Rainfall', 'Max Temp', 'Min Temp', 
                    'Humidity', 'Sunshine', 'Wind Speed', 'Soil pH', 'Nitrogen', 
                    'Phosphorus', 'Potassium', 'Organic Carbon', 'Soil Moisture', 'EC'
                ]

                for col in num_cols:
                    noise_lvl = 0.05 if col in ['Rainfall', 'Area'] else 0.02
                    noise = np.random.normal(0, abs(base_row[col]) * noise_lvl)
                    base_row[col] = max(0, base_row[col] + noise)

                # Heuristic Yield variation with trend
                base_row['Yield'] = base_row['Yield'] * trend_factor * (1 + np.random.normal(0, 0.03))
                base_row['Production'] = (base_row['Yield'] * base_row['Area']) / 1000
                base_row['Is_Expanded'] = True

                expanded_df.append(pd.DataFrame([base_row]))

    df = pd.concat(expanded_df, ignore_index=True)
else:
    df['Is_Expanded'] = False

# Sort by Crop and Year to maintain temporal structure for sequence building
df = df.sort_values(['Crop', 'Crop_Year']).reset_index(drop=True)

# Save as the main dataset
df.to_csv('DK_Final_Research_Dataset.csv', index=False)
print(f"Final expanded dataset saved with {len(df)} records for {len(df['Crop'].unique())} crops.")
