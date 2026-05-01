# ============================================================
#  predict.py — Standalone Crop Yield Predictor
#  Usage: python predict.py
# ============================================================

import numpy as np
import os, sys
import logging
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from feature_pipeline import TRAIN_FEATURES, build_feature_row, load_feature_metadata

# Must define the same wrapper class for joblib to load it
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='LSTM', epochs=50, batch_size=64, n_features=len(TRAIN_FEATURES)):
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
                self.model = tf.keras.models.load_model(tmp.name, compile=False)
            os.unlink(tmp.name)
            del self.model_bytes
    def predict(self, X):
        X_res = np.array(X).reshape((-1, 1, self.n_features))
        return self.model.predict(X_res, verbose=0).flatten()

CROPS_DEFAULT = ['Rice', 'Coconut', 'Cocoa', 'Arecanut', 'Banana', 'Black Pepper', 'Cashewnut', 'Sweet Potato']
_le_path = os.path.join('outputs', 'preprocessors', 'label_encoder.joblib')
if os.path.exists(_le_path):
    try:
        _le = joblib.load(_le_path)
        encoder_crops = [str(c) for c in _le.classes_]
        CROPS = encoder_crops + [c for c in CROPS_DEFAULT if c not in encoder_crops]
    except Exception:
        CROPS = CROPS_DEFAULT
else:
    CROPS = CROPS_DEFAULT
MODEL_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'Attention-LSTM']
CROP_BASE   = {
    'Rice': 2.4,
    'Coconut': 10.0,
    'Cocoa': 0.9,
    'Arecanut': 7.2,
    'Banana': 9.1,
    'Black Pepper': 2.6,
    'Black pepper': 2.6,
    'Cashewnut': 1.0,
    'Sweet Potato': 20.0,
    'Mango': 8.0,
}
CROP_UNITS  = {
    'Rice': 'tonnes/ha',
    'Coconut': '1000s nuts/ha',
    'Cocoa': 'tonnes/ha',
    'Arecanut': 'tonnes/ha',
    'Banana': 'tonnes/ha',
    'Black Pepper': 'tonnes/ha',
    'Black pepper': 'tonnes/ha',
    'Cashewnut': 'tonnes/ha',
    'Sweet Potato': 'tonnes/ha',
    'Mango': 'tonnes/ha',
}
CANONICAL_CROPS = {}
SEQ_LEN = 6

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


FEATURES_PATH = os.path.join('outputs', 'preprocessors', 'features.json')
if os.path.exists(FEATURES_PATH):
    FEATURE_ORDER, FEATURE_DEFAULTS = load_feature_metadata(FEATURES_PATH)
else:
    FEATURE_ORDER, FEATURE_DEFAULTS = TRAIN_FEATURES, {}

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

def load_keras_model(model_path, model_name):
    logger.info("Loading model: %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Loaded model: %s", model_name)
    return model


def predict(crop, model_name, weather, soil, area):
    # Paths
    base_path = os.path.join('outputs', 'preprocessors')
    le_path   = os.path.join(base_path, 'label_encoder.joblib')
    model_path = os.path.join('outputs', 'models', f"{model_name}.joblib")
    keras_path = os.path.join('outputs', 'models', f"{model_name}.keras")
    keras_best_path = os.path.join('outputs', 'models', f"{model_name}_best.keras")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found: {le_path}")
    if not (os.path.exists(keras_best_path) or os.path.exists(keras_path) or os.path.exists(model_path)):
        raise FileNotFoundError(
            f"No model found for {model_name}. Checked: {keras_best_path}, {keras_path}, {model_path}"
        )

    try:
        le = joblib.load(le_path)
        if os.path.exists(keras_best_path) or os.path.exists(keras_path):
            chosen_path = keras_best_path if os.path.exists(keras_best_path) else keras_path
            model = load_keras_model(chosen_path, model_name)
        else:
            logger.info("Loading joblib model: %s", model_path)
            model = joblib.load(model_path)

        scaler_path = os.path.join(base_path, 'scaler.joblib')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        crop_key = CANONICAL_CROPS.get(crop, crop)
        if crop_key not in le.classes_:
            raise ValueError(
                f"Crop '{crop_key}' not found in trained label encoder. Retrain models with this crop."
            )
        crop_enc = le.transform([crop_key])[0]
        weather_map = {
            'Rainfall': weather['rainfall'],
            'Max Temp': weather['max_temp'],
            'Min Temp': weather['min_temp'],
            'Humidity': weather['humidity'],
            'Sunshine': weather['sunshine'],
            'Wind Speed': weather['wind'],
        }
        soil_map = {
            'Soil pH': soil['ph'],
            'Nitrogen': soil['nitrogen'],
            'Phosphorus': soil['phosphorus'],
            'Potassium': soil['potassium'],
            'Organic Carbon': soil['organic_carbon'],
            'Soil Moisture': soil['moisture'],
            'EC': soil['ec'],
        }
        features = build_feature_row(
            weather_map, soil_map, area, crop_enc, defaults=FEATURE_DEFAULTS
        )

        if scaler:
            features = scaler.transform(features)

        input_3d = np.repeat(features.reshape(1, 1, -1), SEQ_LEN, axis=1)
        raw_pred = model.predict(input_3d, verbose=0)
        pred_log = float(np.squeeze(raw_pred))
        pred_yield = float(np.expm1(pred_log))

        return max(0.0, pred_yield), 'model'
    except Exception as e:
        logger.error("Model load/predict failed for %s: %s", model_name, e)
        raise

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
        print("="*60)

        # All-model comparison using actual model outputs
        print("\n  -- All Models Comparison ---------------")
        for m in MODEL_NAMES:
            try:
                y, _ = predict(crop, m, weather, soil, area)
                tag = " ← selected" if m == model_name else ""
                bar_len = int(y / (max(CROP_BASE.values()) / 30))
                bar = "#" * max(1, bar_len)
                print(f"  {m:<14} {y:>8,.0f} {CROP_UNITS[crop]}  {bar}{tag}")
            except Exception as e:
                print(f"  {m:<14} unavailable ({e})")

        again = input("\n  Predict again? (y/n): ").strip().lower()
        if again != 'y': break

    print("\n  Thank you for using Crop Yield Predictor!\n")

if __name__ == '__main__':
    main()
