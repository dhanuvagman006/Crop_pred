"""Research Data Augmentation for Crop Yield Prediction.

This script synthesizes a large-scale research dataset for Dakshina Kannada (DK).
It ensures scientifically plausible relationships between weather, soil, and yield 
to provide 'Research Grade' results for ML modeling.
"""

import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

CROPS = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
EXPANDED_SAMPLES_PER_CROP = 2000

def generate_research_dataset():
    print("Generating Research Grade Dataset...")
    
    all_data = []
    
    # Define Base Characteristics per Crop (tonnes/ha or 1000s nuts/ha)
    # Includes sensitivity to Rainfall, Temp, and Soil
    crop_profiles = {
        'Rice':         {'base': 2.4,  'rain_opt': 3500, 'temp_opt': 27, 'soil_ph_opt': 6.5},
        'Coconut':      {'base': 10.0, 'rain_opt': 2500, 'temp_opt': 28, 'soil_ph_opt': 6.0},
        'Arecanut':     {'base': 7.2,  'rain_opt': 4000, 'temp_opt': 26, 'soil_ph_opt': 5.5},
        'Banana':       {'base': 9.1,  'rain_opt': 2000, 'temp_opt': 30, 'soil_ph_opt': 7.0},
        'Black pepper': {'base': 2.6,  'rain_opt': 3000, 'temp_opt': 25, 'soil_ph_opt': 6.0},
        'Cocoa':        {'base': 0.6,  'rain_opt': 2000, 'temp_opt': 26, 'soil_ph_opt': 6.5},
        'Cashewnut':    {'base': 0.5,  'rain_opt': 1500, 'temp_opt': 32, 'soil_ph_opt': 5.0},
        'Mango':        {'base': 5.0,  'rain_opt': 1200, 'temp_opt': 33, 'soil_ph_opt': 6.5},
    }

    for crop in CROPS:
        print(f"  Processing {crop}...")
        profile = crop_profiles[crop]
        
        for _ in range(EXPANDED_SAMPLES_PER_CROP):
            # 1. Synthesize Features with natural variation
            year = np.random.randint(2000, 2025)
            rainfall = np.random.normal(profile['rain_opt'], 500)
            max_temp = np.random.normal(profile['temp_opt'] + 2, 2)
            min_temp = np.random.normal(profile['temp_opt'] - 4, 2)
            avg_temp = (max_temp + min_temp) / 2
            humidity = np.clip(np.random.normal(75, 10), 40, 95)
            sunshine = np.clip(np.random.normal(6, 1.5), 2, 10)
            wind_speed = np.clip(np.random.normal(8, 2), 2, 20)
            
            soil_ph = np.clip(np.random.normal(profile['soil_ph_opt'], 0.5), 4.5, 8.5)
            nitrogen = np.random.normal(150, 30)
            phosphorus = np.random.normal(40, 10)
            potassium = np.random.normal(200, 50)
            org_carbon = np.random.normal(0.6, 0.15)
            soil_moist = np.clip(np.random.normal(30, 5), 10, 50)
            ec = np.random.normal(0.5, 0.1)
            
            area = np.random.uniform(500, 5000)
            
            # 2. Calculate Research-Grade Yield based on Feature-Response Functions
            # Yield = Base * Rain_Factor * Temp_Factor * Soil_Factor + Noise
            
            # Quadratic response to rainfall (penalty for being too far from optimum)
            rain_factor = 1.0 - 0.2 * ((rainfall - profile['rain_opt']) / 1000)**2
            # Linear/Quadratic response to temp
            temp_factor = 1.0 - 0.15 * ((avg_temp - profile['temp_opt']) / 5)**2
            # Soil pH factor
            soil_factor = 1.0 - 0.1 * ((soil_ph - profile['soil_ph_opt']) / 1.0)**2
            
            # Nutrient factor (N, P, K)
            nutrient_factor = 0.9 + 0.1 * (nitrogen/150 + phosphorus/40 + potassium/200) / 3
            
            factors = rain_factor * temp_factor * soil_factor * nutrient_factor
            factors = np.clip(factors, 0.5, 1.5) # Keep yield within 50% - 150% of base
            
            yield_val = profile['base'] * factors + np.random.normal(0, profile['base'] * 0.05)
            yield_val = max(0.1, yield_val) # Ensure positive yield
            
            production = yield_val * area
            
            all_data.append({
                'Crop': crop,
                'Year': year,
                'Rainfall': round(rainfall, 2),
                'Max Temp': round(max_temp, 2),
                'Min Temp': round(min_temp, 2),
                'Humidity': round(humidity, 2),
                'Sunshine': round(sunshine, 2),
                'Wind Speed': round(wind_speed, 2),
                'Soil pH': round(soil_ph, 2),
                'Nitrogen': round(nitrogen, 2),
                'Phosphorus': round(phosphorus, 2),
                'Potassium': round(potassium, 2),
                'Organic Carbon': round(org_carbon, 2),
                'Soil Moisture': round(soil_moist, 2),
                'EC': round(ec, 2),
                'Area': round(area, 2),
                'Production': round(production, 2),
                'Yield': round(yield_val, 4)
            })

    df = pd.DataFrame(all_data)
    
    # Mix in some real data if exists to anchor it
    if os.path.exists('DK_Real_CropYield.csv'):
        real_df = pd.read_csv('DK_Real_CropYield.csv')
        # Standardize columns
        real_df.columns = [c.strip() for c in real_df.columns]
        if 'Crop_Year' in real_df.columns:
            real_df = real_df.rename(columns={'Crop_Year': 'Year'})
        
        # Filter for our crops
        real_df = real_df[real_df['Crop'].isin(CROPS)]
        
        # Ensure production/yield are present
        if 'Production' in real_df.columns and 'Area' in real_df.columns:
            real_df['Yield'] = real_df['Production'] / real_df['Area']
            real_df = real_df.dropna(subset=['Yield'])
            
            # Select relevant columns
            common_cols = [c for c in df.columns if c in real_df.columns]
            real_subset = real_df[common_cols].copy()
            
            # For missing columns in real data (like soil/weather not in original), fill with mean
            for col in df.columns:
                if col not in real_subset.columns:
                    real_subset[col] = df[df['Crop'].isin(real_subset['Crop'])][col].mean()
            
            df = pd.concat([df, real_subset], ignore_index=True)
            print(f"  Mixed in {len(real_subset)} real records.")

    output_path = 'DK_Final_Research_Dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Success: Research grade dataset saved to {output_path} ({len(df)} records).")

if __name__ == '__main__':
    generate_research_dataset()
