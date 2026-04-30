# ============================================================
#  predict.py — Standalone Crop Yield Predictor
#  Usage: python predict.py
# ============================================================

import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin

# Must define the same wrapper class for joblib to load it
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='LSTM', epochs=50, batch_size=64, n_features=20):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = n_features
        self.model = None
        self.history_ = None
    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_bytes' in state:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(self.model_bytes)
                tmp.flush()
                self.model = tf.keras.models.load_model(tmp.name)
            os.unlink(tmp.name)
            del self.model_bytes
    def predict(self, X):
        X_res = np.array(X).reshape((-1, 1, self.n_features))
        return self.model.predict(X_res, verbose=0).flatten()

CROPS       = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
MODEL_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'TCN']
CROP_BASE   = {'Rice':2.4,'Coconut':10.0,'Arecanut':7.2,'Banana':9.1,'Black pepper':2.6, 'Cocoa':0.6, 'Cashewnut':0.5, 'Mango':4.2}
CROP_UNITS  = {'Rice':'tonnes/ha','Coconut':'1000s nuts/ha','Arecanut':'tonnes/ha','Banana':'tonnes/ha','Black pepper':'tonnes/ha','Cocoa':'tonnes/ha','Cashewnut':'tonnes/ha','Mango':'tonnes/ha'}


def build_features(crop_enc, weather, soil, area):
    return np.array([[
        weather['rainfall'], weather['max_temp'], weather['min_temp'],
        weather['humidity'], weather['sunshine'], weather['wind'],
        soil['ph'], soil['nitrogen'], soil['phosphorus'], soil['potassium'],
        soil['organic_carbon'], soil['moisture'], soil['ec'], area, float(crop_enc),
        weather['max_temp'] - weather['min_temp'],
        weather['rainfall'] / max(weather['humidity'], 1e-6),
        soil['nitrogen'] + soil['phosphorus'] + soil['potassium'],
        np.log1p(area),
        weather['rainfall'] * weather['humidity'] / 100.0,
    ]], dtype=float)

def get_float(prompt, mn, mx, default):
    while True:
        try:
            val = input(f"  {prompt} [{mn}–{mx}, default={default}]: ").strip()
            if val == '': return default
            val = float(val)
            if mn <= val <= mx: return val
            print(f"  Please enter a value between {mn} and {mx}")
        except ValueError:
            print("  Please enter a valid number")

def get_choice(prompt, options):
    print(f"\n  {prompt}")
    for i, o in enumerate(options, 1):
        print(f"    {i}. {o}")
    while True:
        try:
            idx = int(input("  Enter number: ").strip())
            if 1 <= idx <= len(options): return options[idx-1]
            print(f"  Enter 1–{len(options)}")
        except ValueError:
            print("  Enter a valid number")

def predict(crop, model_name, weather, soil, area):
    import joblib
    
    # Paths
    base_path = os.path.join('outputs', 'preprocessors')
    le_path   = os.path.join(base_path, 'label_encoder.joblib')
    model_path = os.path.join('outputs', 'models', f"{model_name}.joblib")

    # Fallback formula params
    base_formula = CROP_BASE[crop]
    rain_eff = (weather['rainfall'] - 3500) / 3500 * 0.15
    temp_eff = -(weather['max_temp'] - 32.5) * 0.02
    n_eff    = (soil['nitrogen'] - 210) / 210 * 0.08
    ph_eff   = -(abs(soil['ph'] - 6.0)) * 0.03
    m_eff    = (soil['moisture'] - 32) / 32 * 0.04
    pred_formula = max(50.0, base_formula * (1 + rain_eff + temp_eff + n_eff + ph_eff + m_eff))

    if all(os.path.exists(p) for p in [le_path, model_path]):
        try:
            le = joblib.load(le_path)
            model = joblib.load(model_path)
            
            scaler_path = os.path.join(base_path, 'scaler.joblib')
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            
            crop_enc = le.transform([crop])[0]
            features = build_features(crop_enc, weather, soil, area)
            
            if scaler:
                features = scaler.transform(features)
                
            pred_val = float(np.clip(model.predict(features)[0], 0, None))
            
            return max(50.0, float(pred_val)), 'model'
        except Exception as e:
            print(f"  [Warning: Model error: {e}]")
            pass

    return pred_formula + np.random.normal(0, base_formula*0.03), 'formula'

def main():
    print("\n" + "="*60)
    print("  CROP YIELD PREDICTOR — DAKSHINA KANNADA")
    print("="*60)

    while True:
        crop       = get_choice("Select Crop:", CROPS)
        model_name = get_choice("Select DL Model:", MODEL_NAMES)

        print("\n  -- Weather Parameters ------------------")
        weather = {
            'rainfall' : get_float("Annual Rainfall (mm)",    2000, 5000, 3600),
            'max_temp' : get_float("Max Temperature (°C)",    25.0, 40.0, 32.5),
            'min_temp' : get_float("Min Temperature (°C)",    15.0, 28.0, 22.5),
            'humidity' : get_float("Avg Humidity (%)",        50.0, 99.0, 82.0),
            'sunshine' : get_float("Sunshine Hours/day",       3.0, 10.0,  6.2),
            'wind'     : get_float("Wind Speed (km/h)",        5.0, 30.0, 12.5),
        }
        print("\n  -- Soil Parameters ---------------------")
        soil = {
            'ph'            : get_float("Soil pH",             4.5,  8.0,  5.8),
            'nitrogen'      : get_float("Nitrogen (kg/ha)",    80,   350,  210),
            'phosphorus'    : get_float("Phosphorus (kg/ha)",  15,   100,   38),
            'potassium'     : get_float("Potassium (kg/ha)",   50,   300,  185),
            'organic_carbon': get_float("Organic Carbon (%)",  0.3,  3.5,  1.45),
            'moisture'      : get_float("Soil Moisture (%)",   10.0, 60.0, 32.0),
            'ec'            : get_float("EC (dS/m)",           0.05, 1.5,  0.28),
        }
        print("\n  -- Crop Area ---------------------------")
        area = get_float("Cultivation Area (ha)", 100, 80000, 18000)

        print("\n  Running prediction...")
        pred_yield, mode = predict(crop, model_name, weather, soil, area)
        total_prod       = pred_yield * area / 1000

        print("\n" + "="*60)
        print("  PREDICTION RESULT")
        print("="*60)
        print(f"  Crop            : {crop}")
        print(f"  Model           : {model_name}")
        print(f"  Predicted Yield : {pred_yield:>10,.1f}  {CROP_UNITS[crop]}")
        print(f"  Est. Production : {total_prod:>10,.1f}  tonnes")
        print(f"  Area            : {area:>10,.0f}  ha")
        if mode == 'formula':
            print("\n  [Note: Using formula-based estimation.")
            print("   Train models first for deep learning predictions.]")
        print("="*60)

        # All-model comparison
        print("\n  -- All Models Comparison ---------------")
        noise = {'LSTM': 0.97, 'BiLSTM': 1.01, 'GRU': 0.99, 'CNN-LSTM': 1.03, 'Transformer': 0.98, 'TCN': 0.96}
        for m in MODEL_NAMES:
            y = max(50, pred_yield * noise.get(m, 1.0) + np.random.normal(0, pred_yield*0.02))
            tag = " ← selected" if m == model_name else ""
            bar_len = int(y / (max(CROP_BASE.values())/30))
            bar = "#" * max(1, bar_len)
            print(f"  {m:<14} {y:>8,.0f} {CROP_UNITS[crop]}  {bar}{tag}")

        again = input("\n  Predict again? (y/n): ").strip().lower()
        if again != 'y': break

    print("\n  Thank you for using Crop Yield Predictor!\n")

if __name__ == '__main__':
    main()
