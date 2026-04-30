# ============================================================
#  Crop Yield Prediction — Dakshina Kannada
#  6 Deep Learning Models: LSTM | BiLSTM | GRU | CNN-LSTM | Transformer | TCN
# ============================================================

import os, time, warnings, joblib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
import argparse
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D,
    MaxPooling1D, Flatten, Input, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add,
    BatchNormalization, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

# CLI for quick runs and reproducibility
parser = argparse.ArgumentParser(description='Train crop yield models')
parser.add_argument('--quick', action='store_true', help='Run a quick dry-run (fewer epochs, smaller data)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
CFG = {
    'data_path'   : 'DK_Final_Research_Dataset.csv',
    'seq_len'     : 5,          # Increased for better temporal context
    'test_size'   : 0.15,
    'val_size'    : 0.15,
    'epochs'      : 200,        # Optimized epochs
    'batch_size'  : 32,         # Smaller batch for better convergence
    'lr'          : 0.001,      # Lower initial LR for stability
    'patience'    : 40,
    'output_dir'  : 'outputs',
    'model_dir'   : 'outputs/models',
}

if args.quick:
    CFG['epochs'] = 3
    CFG['batch_size'] = 8
    CFG['patience'] = 5

os.makedirs(CFG['output_dir'], exist_ok=True)
os.makedirs(CFG['model_dir'],  exist_ok=True)
os.makedirs(os.path.join(CFG['output_dir'], 'preprocessors'), exist_ok=True)

CROPS        = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
MODEL_NAMES  = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'TCN']
COLORS       = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']

FEATURE_COLS = [
    'Annual_Rainfall_mm', 'Max_Temp_C', 'Min_Temp_C', 'Avg_Humidity_pct',
    'Sunshine_Hours_day', 'Wind_Speed_kmh',
    'Soil_pH', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha',
    'Organic_Carbon_pct', 'Soil_Moisture_pct', 'EC_dSm',
    'Area_ha', 'Crop_encoded'
]
TARGET_COL = 'Yield_kg_ha'

print("=" * 65)
print("   CROP YIELD PREDICTION — DAKSHINA KANNADA")
print("   6 Deep Learning Models Training Pipeline (8 Crops)")
print("=" * 65)

# ──────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
print("\n[1/6] Loading & Preprocessing Dataset...")

df = pd.read_csv(CFG['data_path'])

col_rename = {
    'Crop_Year': 'Year',
    'Area': 'Area_ha', 'Production': 'Production_tonnes',
    'Yield': 'Yield_kg_ha', 'Rainfall': 'Annual_Rainfall_mm',
    'Max Temp': 'Max_Temp_C', 'Min Temp': 'Min_Temp_C',
    'Humidity': 'Avg_Humidity_pct', 'Sunshine': 'Sunshine_Hours_day',
    'Wind Speed': 'Wind_Speed_kmh', 'Soil pH': 'Soil_pH',
    'Nitrogen': 'Nitrogen_kg_ha', 'Phosphorus': 'Phosphorus_kg_ha',
    'Potassium': 'Potassium_kg_ha', 'Organic Carbon': 'Organic_Carbon_pct',
    'Soil Moisture': 'Soil_Moisture_pct', 'EC': 'EC_dSm',
}
df.rename(columns=col_rename, inplace=True)

le = LabelEncoder()
df['Crop_encoded'] = le.fit_transform(df['Crop'])
df = df.sort_values(['Crop', 'Year']).reset_index(drop=True)

# Exclude synthesized/expanded rows if present to avoid biased evaluation
for flag_col in ['Is_Synthesized', 'Is_Expanded']:
    if flag_col in df.columns:
        df = df[~df[flag_col].astype(bool)].reset_index(drop=True)

joblib.dump(le, os.path.join(CFG['output_dir'], 'preprocessors', 'label_encoder.joblib'))

def build_sequences(data_X, data_y, seq_len):
    X, y = [], []
    for i in range(len(data_X) - seq_len):
        X.append(data_X[i:i+seq_len])
        y.append(data_y[i+seq_len])
    return np.array(X), np.array(y)

def prepare_data(crop_name):
    cdf = df[df['Crop'] == crop_name].copy()
    if len(cdf) <= CFG['seq_len'] + 1:
        return np.empty((0, CFG['seq_len'], len(FEATURE_COLS))), np.empty((0,)), np.empty((0, CFG['seq_len'], len(FEATURE_COLS))), np.empty((0,)), np.empty((0, CFG['seq_len'], len(FEATURE_COLS))), np.empty((0,)), None, None

    # Use raw features and years to build sequences first (no leakage)
    X_raw = cdf[FEATURE_COLS].values.astype(float)
    y_raw = cdf[TARGET_COL].values.astype(float).reshape(-1, 1)

    # Build raw sequences (unscaled)
    X_seq_raw, y_seq_raw = build_sequences(X_raw, y_raw, CFG['seq_len'])

    # Year corresponding to each target in the sequence (y index is offset by seq_len)
    y_years = cdf['Year'].values[CFG['seq_len']:]

    # Determine year-based splits to avoid temporal leakage
    unique_years = sorted(np.unique(y_years))
    n_test_years = max(1, int(len(unique_years) * CFG['test_size'] + 0.5))
    n_val_years = max(1, int(len(unique_years) * CFG['val_size'] + 0.5))

    test_years = unique_years[-n_test_years:]
    val_years = unique_years[-(n_test_years + n_val_years):-n_test_years] if len(unique_years) > n_test_years else []

    test_mask = np.isin(y_years, test_years)
    val_mask = np.isin(y_years, val_years) if len(val_years) > 0 else np.zeros_like(test_mask, dtype=bool)
    train_mask = ~(test_mask | val_mask)

    X_train_raw = X_seq_raw[train_mask]
    y_train_raw = y_seq_raw[train_mask].reshape(-1, 1)
    X_val_raw = X_seq_raw[val_mask]
    y_val_raw = y_seq_raw[val_mask].reshape(-1, 1)
    X_test_raw = X_seq_raw[test_mask]
    y_test_raw = y_seq_raw[test_mask].reshape(-1, 1)

    # Fit scalers on training data only (avoid leakage)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit scaler_X over all time-steps of training sequences
    n_feat = X_train_raw.shape[2]
    scaler_X.fit(X_train_raw.reshape(-1, n_feat))
    scaler_y.fit(y_train_raw)

    # Transform datasets
    def transform_X(Xr):
        if Xr.size == 0:
            return np.empty((0, CFG['seq_len'], n_feat))
        return scaler_X.transform(Xr.reshape(-1, n_feat)).reshape(-1, CFG['seq_len'], n_feat)

    X_train = transform_X(X_train_raw)
    X_val = transform_X(X_val_raw)
    X_test = transform_X(X_test_raw)

    y_train = scaler_y.transform(y_train_raw).flatten() if y_train_raw.size else np.empty((0,))
    y_val = scaler_y.transform(y_val_raw).flatten() if y_val_raw.size else np.empty((0,))
    y_test = scaler_y.transform(y_test_raw).flatten() if y_test_raw.size else np.empty((0,))

    # Shuffle training data only
    if len(X_train) > 0:
        idx = np.arange(len(X_train))
        np.random.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y

# ──────────────────────────────────────────────
# 4. MODEL ARCHITECTURES
# ──────────────────────────────────────────────
def get_callbacks(model_name, crop):
    os.makedirs(os.path.join(CFG['model_dir'], 'checkpoints'), exist_ok=True)
    cp_path = os.path.join(CFG['model_dir'], 'checkpoints', f'{model_name}_{crop.replace(" ","_")}_best.keras')
    return [
        EarlyStopping(monitor='val_loss', patience=CFG['patience'], restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ModelCheckpoint(cp_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

def build_lstm(seq_len, n_features):
    m = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True), BatchNormalization(),
        LSTM(64), BatchNormalization(),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(32, activation='relu'), Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

def build_bilstm(seq_len, n_features):
    m = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(128, return_sequences=True)), BatchNormalization(),
        Bidirectional(LSTM(64)), BatchNormalization(),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(32, activation='relu'), Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

def build_gru(seq_len, n_features):
    m = Sequential([
        Input(shape=(seq_len, n_features)),
        GRU(128, return_sequences=True), BatchNormalization(), Dropout(0.1),
        GRU(64), BatchNormalization(), Dropout(0.1),
        Dense(32, activation='relu'), Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

def build_cnn_lstm(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x   = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inp)
    x   = BatchNormalization()(x)
    x   = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    x   = LSTM(64)(x)
    x   = Dropout(0.1)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    m   = Model(inp, out, name='CNN_LSTM'); m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

def build_transformer(seq_len, n_features, num_heads=4, ff_dim=128):
    inp = Input(shape=(seq_len, n_features))
    x   = Dense(64)(inp)
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x1  = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))
    ff  = Dense(ff_dim, activation='relu')(x1); ff = Dense(64)(ff)
    x2  = LayerNormalization(epsilon=1e-6)(Add()([x1, ff]))
    x3  = GlobalAveragePooling1D()(x2); x3 = Dropout(0.1)(x3); x3 = Dense(32, activation='relu')(x3); out = Dense(1)(x3)
    m   = Model(inp, out, name='Transformer'); m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

def build_tcn(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(128, kernel_size=2, dilation_rate=1, padding='causal', activation='relu')(inp); x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=2, dilation_rate=2, padding='causal', activation='relu')(x); x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x); x = Dense(64, activation='relu')(x); x = Dropout(0.1)(x); out = Dense(1)(x)
    m = Model(inp, out, name='TCN'); m.compile(optimizer=Adam(learning_rate=CFG['lr'], clipnorm=1.0), loss='huber', metrics=['mae'])
    return m

MODEL_BUILDERS = {'LSTM': build_lstm, 'BiLSTM': build_bilstm, 'GRU': build_gru, 'CNN-LSTM': build_cnn_lstm, 'Transformer': build_transformer, 'TCN': build_tcn}

# ──────────────────────────────────────────────
# 5. METRICS
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {'R2': round(r2, 4), 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}

# ──────────────────────────────────────────────
# 6. TRAINING LOOP
# ──────────────────────────────────────────────
print("\n[3/6] Training All Models on All Crops...")
all_results  = []; all_preds    = {}; train_times  = {}; param_counts = {}; histories    = {}

for model_name in MODEL_NAMES:
    all_preds[model_name]   = {}; train_times[model_name] = {}; histories[model_name]   = {}

for crop in CROPS:
    print(f"\n  Crop: {crop}")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = prepare_data(crop)
    if scaler_X is None or scaler_y is None or X_train.size == 0 or X_test.size == 0:
        print(f"    Skipping {crop}: insufficient data after filtering/splitting")
        continue
    n_feat = X_train.shape[2]

    # Save Scalers for this crop
    joblib.dump(scaler_X, os.path.join(CFG['output_dir'], 'preprocessors', f'scaler_X_{crop.replace(" ","_")}.joblib'))
    joblib.dump(scaler_y, os.path.join(CFG['output_dir'], 'preprocessors', f'scaler_y_{crop.replace(" ","_")}.joblib'))

    for model_name in MODEL_NAMES:
        model_path = os.path.join(CFG['model_dir'], f'{model_name}_{crop.replace(" ","_")}.keras')
        t0 = time.time()
        model = MODEL_BUILDERS[model_name](CFG['seq_len'], n_feat)
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=CFG['epochs'], batch_size=CFG['batch_size'], callbacks=get_callbacks(model_name, crop), verbose=0)
        elapsed = round(time.time() - t0, 2)
        
        # Load best weights if checkpoint exists
        best_path = os.path.join(CFG['model_dir'], 'checkpoints', f'{model_name}_{crop.replace(" ","_")}_best.keras')
        if os.path.exists(best_path):
            model = tf.keras.models.load_model(best_path)
            
        model.save(model_path)
        
        train_times[model_name][crop] = elapsed
        histories[model_name][crop]   = hist.history
        param_counts[model_name]      = model.count_params()

        y_pred_s = model.predict(X_test, verbose=0).flatten()
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).flatten()
        y_true   = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

        all_preds[model_name][crop] = (y_true, y_pred)
        metrics = compute_metrics(y_true, y_pred)
        all_results.append({'Model': model_name, 'Crop': crop, **metrics, 'Train_Time_s': elapsed, 'Params': param_counts[model_name]})
        print(f"    [{model_name:12s}] R²={metrics['R2']:.4f} MAPE={metrics['MAPE']:.1f}% ({elapsed}s)")

results_df = pd.DataFrame(all_results)

# ──────────────────────────────────────────────
# 7. SAVE RESULTS
# ──────────────────────────────────────────────
print("\n[5/6] Saving Results Report...")
avg_perf = results_df.groupby('Model').agg(Avg_R2=('R2','mean'), Avg_RMSE=('RMSE','mean'), Avg_MAE=('MAE','mean'), Avg_MAPE=('MAPE','mean'), Total_Train_Time=('Train_Time_s','sum'), Params=('Params','first')).reset_index()
avg_perf['Composite'] = avg_perf['Avg_R2'] # Simplified
avg_perf = avg_perf.sort_values('Composite', ascending=False)
results_df.to_csv(os.path.join(CFG['output_dir'], 'detailed_metrics.csv'), index=False)
avg_perf.to_csv(os.path.join(CFG['output_dir'], 'model_comparison.csv'), index=False)
print("\n[6/6] Training Complete!\n")
