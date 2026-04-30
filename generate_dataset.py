import pandas as pd
import numpy as np
import os

# Configuration
output_file = 'DK_CropYield_Dataset.csv'
crops = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black Pepper']
years = range(2010, 2025)

# Realistic ranges for Dakshina Kannada
# Rainfall: 3000-4500 mm
# Temp: 22-34 C
# Humidity: 70-90%

data = []

for crop in crops:
    for year in years:
        # Generate weather (correlated roughly)
        rainfall = np.random.uniform(2800, 4800)
        max_temp = np.random.uniform(28, 36)
        min_temp = np.random.uniform(18, 26)
        humidity = np.random.uniform(70, 95)
        sunshine = np.random.uniform(4, 9)
        wind = np.random.uniform(8, 20)
        
        # Soil parameters
        ph = np.random.uniform(5.0, 7.0)
        nitrogen = np.random.uniform(120, 300)
        phosphorus = np.random.uniform(20, 80)
        potassium = np.random.uniform(100, 250)
        org_carbon = np.random.uniform(0.8, 2.5)
        moisture = np.random.uniform(20, 50)
        ec = np.random.uniform(0.1, 0.6)
        
        area = np.random.uniform(1000, 50000)
        
        # Heuristic for yield
        base_yield = {
            'Rice': 2500, 'Coconut': 9000, 'Arecanut': 2000,
            'Banana': 20000, 'Black Pepper': 300
        }[crop]
        
        # Simple effects
        y_eff = (rainfall / 3500) * 1.1 + (nitrogen / 200) * 1.05 + (ph / 6.0) * 1.02
        y_eff += np.random.normal(0, 0.05) # noise
        
        yield_val = base_yield * y_eff
        production = (yield_val * area) / 1000 # tonnes
        
        data.append({
            'Crop': crop,
            'Year': year,
            'Rainfall': rainfall,
            'Max Temp': max_temp,
            'Min Temp': min_temp,
            'Humidity': humidity,
            'Sunshine': sunshine,
            'Wind Speed': wind,
            'Soil pH': ph,
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Organic Carbon': org_carbon,
            'Soil Moisture': moisture,
            'EC': ec,
            'Area': area,
            'Yield': yield_val,
            'Production': production
        })

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"Successfully generated synthetic dataset: {output_file} with {len(df)} records.")
