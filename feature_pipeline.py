import json
from pathlib import Path
import numpy as np

# Shared, ordered feature list used by both training and inference (22 features)
TRAIN_FEATURES = [
    'Rainfall', 'Max Temp', 'Min Temp', 'Humidity', 'Sunshine',
    'Wind Speed', 'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium',
    'Organic Carbon', 'Soil Moisture', 'EC', 'Area', 'crop_encoded',
    'temp_range', 'rain_humidity_ratio', 'npk_sum', 'log_area',
    'rain_humidity_product', 'heat_stress_index', 'soil_ph_deviation'
]


def add_engineered_features(df):
    df = df.copy()
    df['temp_range'] = df['Max Temp'] - df['Min Temp']
    df['heat_stress_index'] = df['Max Temp'] * df['Humidity'] / 100.0
    df['soil_ph_deviation'] = (df['Soil pH'] - 6.5).abs()
    df['rain_humidity_ratio'] = df['Rainfall'] / df['Humidity'].clip(lower=1e-6)
    df['npk_sum'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
    df['log_area'] = np.log1p(df['Area'])
    df['rain_humidity_product'] = df['Rainfall'] * df['Humidity'] / 100.0
    return df


def build_feature_row(weather, soil, area, crop_enc, defaults=None):
    def get_val(src, key):
        if src is not None and key in src and src[key] is not None:
            return src[key]
        if defaults is not None and key in defaults:
            return defaults[key]
        return 0.0

    rainfall = float(get_val(weather, 'Rainfall'))
    max_temp = float(get_val(weather, 'Max Temp'))
    min_temp = float(get_val(weather, 'Min Temp'))
    humidity = float(get_val(weather, 'Humidity'))
    sunshine = float(get_val(weather, 'Sunshine'))
    wind = float(get_val(weather, 'Wind Speed'))

    soil_ph = float(get_val(soil, 'Soil pH'))
    nitrogen = float(get_val(soil, 'Nitrogen'))
    phosphorus = float(get_val(soil, 'Phosphorus'))
    potassium = float(get_val(soil, 'Potassium'))
    organic_carbon = float(get_val(soil, 'Organic Carbon'))
    soil_moisture = float(get_val(soil, 'Soil Moisture'))
    ec = float(get_val(soil, 'EC'))

    area_val = float(area)
    crop_enc_val = float(crop_enc)

    temp_range = max_temp - min_temp
    rain_humidity_ratio = rainfall / max(humidity, 1e-6)
    npk_sum = nitrogen + phosphorus + potassium
    log_area = np.log1p(area_val)
    rain_humidity_product = rainfall * humidity / 100.0
    heat_stress_index = max_temp * humidity / 100.0
    soil_ph_deviation = abs(soil_ph - 6.5)

    ordered = [
        rainfall, max_temp, min_temp, humidity, sunshine,
        wind, soil_ph, nitrogen, phosphorus, potassium,
        organic_carbon, soil_moisture, ec, area_val, crop_enc_val,
        temp_range, rain_humidity_ratio, npk_sum, log_area,
        rain_humidity_product, heat_stress_index, soil_ph_deviation
    ]
    return np.array([ordered], dtype=float)


def save_feature_metadata(path, features, defaults):
    payload = {
        'feature_order': features,
        'defaults': defaults,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_feature_metadata(path):
    meta = json.loads(Path(path).read_text())
    return meta.get('feature_order', []), meta.get('defaults', {})
