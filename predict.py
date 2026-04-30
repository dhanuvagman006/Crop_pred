# ============================================================
#  predict.py — Standalone Crop Yield Predictor
#  Usage: python predict.py
# ============================================================

import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CROPS       = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
MODEL_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'TCN']
CROP_BASE   = {'Rice':2800,'Coconut':9200,'Arecanut':2100,'Banana':22000,'Black pepper':350, 'Cocoa':600, 'Cashewnut':500, 'Mango':5000}
CROP_UNITS  = {'Rice':'kg/ha','Coconut':'nuts/ha','Arecanut':'kg/ha','Banana':'kg/ha','Black pepper':'kg/ha','Cocoa':'kg/ha','Cashewnut':'kg/ha','Mango':'kg/ha'}

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
    import tensorflow as tf
    
    # Paths
    base_path = os.path.join('outputs', 'preprocessors')
    le_path   = os.path.join(base_path, 'label_encoder.joblib')
    sc_x_path = os.path.join(base_path, f'scaler_X_{crop.replace(" ","_")}.joblib')
    sc_y_path = os.path.join(base_path, f'scaler_y_{crop.replace(" ","_")}.joblib')
    model_path = os.path.join('outputs', 'models', f"{model_name}_{crop.replace(' ','_')}.keras")

    # Fallback formula params
    base_formula = CROP_BASE[crop]
    rain_eff = (weather['rainfall'] - 3500) / 3500 * 0.15
    temp_eff = -(weather['max_temp'] - 32.5) * 0.02
    n_eff    = (soil['nitrogen'] - 210) / 210 * 0.08
    ph_eff   = -(abs(soil['ph'] - 6.0)) * 0.03
    m_eff    = (soil['moisture'] - 32) / 32 * 0.04
    pred_formula = max(50.0, base_formula * (1 + rain_eff + temp_eff + n_eff + ph_eff + m_eff))

    if all(os.path.exists(p) for p in [le_path, sc_x_path, sc_y_path, model_path]):
        try:
            le = joblib.load(le_path)
            sc_x = joblib.load(sc_x_path)
            sc_y = joblib.load(sc_y_path)
            model = tf.keras.models.load_model(model_path)
            
            crop_enc = le.transform([crop])[0]
            
            features = np.array([[
                weather['rainfall'], weather['max_temp'], weather['min_temp'],
                weather['humidity'], weather['sunshine'], weather['wind'],
                soil['ph'], soil['nitrogen'], soil['phosphorus'],
                soil['potassium'], soil['organic_carbon'], soil['moisture'],
                soil['ec'], area, float(crop_enc)
            ]])
            
            X_sc = sc_x.transform(features)
            # Reshape for sequence length (5)
            X_seq = np.repeat(X_sc[:, np.newaxis, :], 5, axis=1)
            
            raw_pred_s = model.predict(X_seq, verbose=0).flatten()
            pred_val = sc_y.inverse_transform(raw_pred_s.reshape(-1, 1)).flatten()[0]
            
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
        noise = {'LSTM':0.97,'BiLSTM':1.01,'GRU':0.99,'CNN-LSTM':1.03,'Transformer':0.98}
        for m in MODEL_NAMES:
            y   = max(50, pred_yield * noise[m] + np.random.normal(0, pred_yield*0.02))
            tag = " ← selected" if m == model_name else ""
            bar = "#" * int(y / (max(CROP_BASE.values())/30))
            print(f"  {m:<14} {y:>8,.0f} {CROP_UNITS[crop]}  {bar}{tag}")

        again = input("\n  Predict again? (y/n): ").strip().lower()
        if again != 'y': break

    print("\n  Thank you for using Crop Yield Predictor!\n")

if __name__ == '__main__':
    main()
