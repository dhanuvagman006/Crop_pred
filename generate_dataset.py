import pandas as pd
import numpy as np

np.random.seed(42)

crops = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black Pepper']
base_yields = {
    'Rice': 8500,
    'Coconut': 30000,
    'Arecanut': 6700,
    'Banana': 70000,
    'Black Pepper': 1000
}

years = list(range(1800, 2025))

data = []
for crop in crops:
    base_yield = base_yields[crop]
    for year in years:
        rainfall = np.random.uniform(3000, 4500)
        max_temp = np.random.uniform(28, 38)
        min_temp = np.random.uniform(20, 26)
        humidity = np.random.uniform(60, 95)
        sunshine = np.random.uniform(4, 9)
        wind_speed = np.random.uniform(10, 25)
        
        soil_ph = np.random.uniform(5.0, 7.5)
        nitrogen = np.random.uniform(150, 300)
        phosphorus = np.random.uniform(20, 80)
        potassium = np.random.uniform(100, 250)
        organic_carbon = np.random.uniform(1.0, 3.5)
        soil_moisture = np.random.uniform(20, 50)
        ec = np.random.uniform(0.1, 0.8)
        
        area = np.random.uniform(1000, 50000)
        
        # Perfect formula with a tiny bit of noise so it's learnable
        rain_eff = 0.3 * np.sin(rainfall / 1000)
        temp_eff = -0.1 * abs(max_temp - 32)
        n_eff = 0.1 * (nitrogen / 200)
        ph_eff = -0.05 * abs(soil_ph - 6.5)
        m_eff = 0.05 * (soil_moisture / 30)
        
        # Add a time-trend to simulate technological improvement
        trend = (year - 1800) * 0.001
        
        multiplier = 1.0 + rain_eff + temp_eff + n_eff + ph_eff + m_eff + trend
        
        # 2% random noise
        noise = np.random.normal(1.0, 0.02)
        
        final_yield = base_yield * multiplier * noise
        production = final_yield * area
        
        data.append([
            crop, year, rainfall, max_temp, min_temp, humidity, sunshine, wind_speed,
            soil_ph, nitrogen, phosphorus, potassium, organic_carbon, soil_moisture, ec,
            area, final_yield, production
        ])

columns = [
    'Crop', 'Year', 'Rainfall', 'Max Temp', 'Min Temp', 'Humidity', 'Sunshine', 'Wind Speed',
    'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic Carbon', 'Soil Moisture', 'EC',
    'Area', 'Yield', 'Production'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv('DK_CropYield_Dataset.csv', index=False)
print("Generated DK_CropYield_Dataset.csv with", len(df), "rows.")
