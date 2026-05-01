"""
High-Accuracy Crop Yield Prediction — 6 DL Models
Target: R² >= 0.88 on all models
"""
from __future__ import annotations
import os, time, warnings, concurrent.futures
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional,
    Conv1D, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Multiply, Softmax, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from feature_pipeline import TRAIN_FEATURES as SHARED_FEATURES, add_engineered_features, save_feature_metadata

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path('outputs')
MODEL_DIR = BASE / 'models'
PREPROC_DIR = BASE / 'preprocessors'
HIST_DIR = BASE / 'histories'
PRED_DIR = BASE / 'predictions'
for d in [MODEL_DIR, PREPROC_DIR, HIST_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_PATH = 'DK_CropYield_Dataset.csv'
SEQ_LEN = 5
MODEL_NAMES = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'Attention-LSTM']

# ── Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['Crop'])
    joblib.dump(le, PREPROC_DIR / 'label_encoder.joblib')

    df = add_engineered_features(df)

    for crop in df['Crop'].unique():
        mask = df['Crop'] == crop
        lo, hi = df.loc[mask, 'Yield'].quantile([0.02, 0.98])
        df.loc[mask, 'Yield'] = df.loc[mask, 'Yield'].clip(lo, hi)

    for col in ['yield_lag1', 'yield_lag2', 'yield_rolling3']:
        df[col] = np.nan

    for crop in df['Crop'].unique():
        mask = df['Crop'] == crop
        grp = df.loc[mask].sort_values('Year')
        df.loc[grp.index, 'yield_lag1'] = grp['Yield'].shift(1).values
        df.loc[grp.index, 'yield_lag2'] = grp['Yield'].shift(2).values
        df.loc[grp.index, 'yield_rolling3'] = grp['Yield'].rolling(3).mean().values
        gm = grp['Yield'].mean()
        df.loc[grp.index, 'yield_lag1'] = df.loc[grp.index, 'yield_lag1'].fillna(gm)
        df.loc[grp.index, 'yield_lag2'] = df.loc[grp.index, 'yield_lag2'].fillna(gm)
        df.loc[grp.index, 'yield_rolling3'] = df.loc[grp.index, 'yield_rolling3'].fillna(gm)

    for col in ['yield_lag1', 'yield_lag2', 'yield_rolling3']:
        df[col] = np.log1p(df[col])

    return df

TRAIN_FEATURES = SHARED_FEATURES

def make_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = SEQ_LEN):
    X_list, y_list, crops_list, years_list = [], [], [], []
    for crop in df['Crop'].unique():
        sub = df[df['Crop'] == crop].sort_values('Year')
        feats = sub[feature_cols].values
        targets = sub['Yield'].values
        yrs = sub['Year'].values
        for i in range(len(sub) - seq_len):
            X_list.append(feats[i:i + seq_len])
            y_list.append(targets[i + seq_len])
            crops_list.append(crop)
            years_list.append(yrs[i + seq_len])
    return (np.array(X_list), np.array(y_list),
            np.array(crops_list), np.array(years_list))

# ── Model builders ─────────────────────────────────────────────────────────
def build_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = LSTM(32, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(16)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    out = Dense(1)(x)
    return Model(inp, out, name='LSTM')

def build_bilstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = Bidirectional(LSTM(32, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(16))(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    out = Dense(1)(x)
    return Model(inp, out, name='BiLSTM')

def build_gru(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = GRU(32, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = GRU(16)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    out = Dense(1)(x)
    return Model(inp, out, name='GRU')

def build_cnn_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = Conv1D(32, kernel_size=min(3, seq_len), padding='same', activation='relu')(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    out = Dense(1)(x)
    return Model(inp, out, name='CNN-LSTM')

def build_transformer(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = Dense(32)(inp)
    attn1 = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = LayerNormalization()(x + attn1)
    ff1 = Dense(32, activation='relu')(x)
    x = LayerNormalization()(x + ff1)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer='l2')(x)
    out = Dense(1)(x)
    return Model(inp, out, name='Transformer')

def build_attention_lstm(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x = LSTM(32, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    score = Dense(1, use_bias=False)(x)
    weights = Softmax(axis=1)(score)
    context = Multiply()([x, weights])
    context = Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)
    out = Dense(16, activation='relu', kernel_regularizer='l2')(context)
    out = Dense(1)(out)
    model = Model(inp, out, name='Attention-LSTM')
    attn_model = Model(inp, weights, name='Attention-LSTM-weights')
    return model, attn_model

BUILDERS = {
    'LSTM': build_lstm,
    'BiLSTM': build_bilstm,
    'GRU': build_gru,
    'CNN-LSTM': build_cnn_lstm,
    'Transformer': build_transformer,
}

# ── Task for Parallel Execution ───────────────────────────────────────────
def train_single_model_task(name, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, crops_test, n_feat):
    # Set TF threading config for this process
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    t0 = time.time()
    tf.keras.backend.clear_session()
    
    if name == 'Attention-LSTM':
        model, attn_model = build_attention_lstm(SEQ_LEN, n_feat)
    else:
        model = BUILDERS[name](SEQ_LEN, n_feat)
    
    ckpt_path = str(MODEL_DIR / f'{name}_best.keras')
    cbs = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-7),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    ]
    
    opt = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=0.5), metrics=['mae'])
    
    h = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), 
                  epochs=300, batch_size=16, callbacks=cbs, verbose=0)
    
    elapsed = round(time.time() - t0, 2)
    epochs_run = len(h.history['loss'])
    
    # Save
    model.save(str(MODEL_DIR / f'{name}.keras'))
    if name == 'Attention-LSTM':
        attn_model.save(str(MODEL_DIR / 'Attention-LSTM-weights.keras'))
        
    pd.DataFrame({
        'epoch': list(range(1, epochs_run + 1)),
        'loss': h.history['loss'],
        'val_loss': h.history.get('val_loss', [np.nan] * epochs_run),
    }).to_csv(HIST_DIR / f'{name}_history.csv', index=False)
    
    # Predict & Metrics
    y_pred_log = model.predict(X_test_s, verbose=0).flatten()
    y_pred = np.expm1(y_pred_log).clip(0)
    y_actual = np.expm1(y_test)
    
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-9))) * 100
    
    # Per-crop
    detailed = []
    for crop in sorted(np.unique(crops_test)):
        mask = crops_test == crop
        ya, yp = y_actual[mask], y_pred[mask]
        cr2 = r2_score(ya, yp) if len(ya) > 1 else np.nan
        detailed.append({
            'Model': name, 'Crop': crop, 'R2': round(float(cr2), 4),
            'MAE': round(float(mean_absolute_error(ya, yp)), 2),
            'RMSE': round(float(np.sqrt(mean_squared_error(ya, yp))), 2),
            'MAPE': round(float(np.mean(np.abs((ya - yp) / (ya + 1e-9))) * 100), 2)
        })
        pd.DataFrame({'Actual': ya, 'Predicted': yp, 'Model': name, 'Crop': crop}).to_csv(
            PRED_DIR / f'{name}_{crop.replace(" ", "_")}_predictions.csv', index=False
        )

    summary = {
        'Model': name, 'R2': round(r2, 4), 'RMSE': round(rmse, 2),
        'MAE': round(mae, 2), 'MAPE': round(mape, 2),
        'Epochs': epochs_run, 'Total_Train_Time': elapsed
    }
    
    print(f"Finished {name} in {elapsed}s (R2={r2:.4f})")
    return summary, detailed, y_pred

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print('='*70)
    print('Parallel Crop Yield Training — 6 DL Models')
    print('='*70)

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    avail = [c for c in TRAIN_FEATURES if c in df.columns]
    
    years = sorted(df['Year'].unique())
    n = len(years)
    test_yrs = years[int(n * 0.85):]
    val_yrs = years[int(n * 0.70):int(n * 0.85)]
    train_yrs = years[:int(n * 0.70)]
    
    selected = [c for c in TRAIN_FEATURES if c in df.columns]
    if len(selected) != len(TRAIN_FEATURES):
        missing = sorted(set(TRAIN_FEATURES) - set(selected))
        raise ValueError(f"Missing required features for training/inference: {missing}")
    joblib.dump(selected, PREPROC_DIR / 'selected_features.joblib')
    defaults = df[selected].mean(numeric_only=True).to_dict()
    save_feature_metadata(PREPROC_DIR / 'features.json', selected, defaults)
    print(f"Training features ({len(selected)}): {selected}")
    X, y_raw, crops_arr, years_arr = make_sequences(df, selected, SEQ_LEN)
    y = np.log1p(y_raw)
    
    train_val_mask = np.isin(years_arr, train_yrs + val_yrs)
    X_tv, y_tv = X[train_val_mask], y[train_val_mask]
    
    tscv = TimeSeriesSplit(n_splits=2)
    for tr_idx, val_idx in tscv.split(X_tv): pass
    X_train, y_train = X_tv[tr_idx], y_tv[tr_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]
    
    te_mask = np.isin(years_arr, test_yrs)
    X_test, y_test = X[te_mask], y[te_mask]
    crops_test = crops_arr[te_mask]
    
    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    joblib.dump(scaler, PREPROC_DIR / 'scaler.joblib')
    
    def scale(arr):
        sh = arr.shape
        return scaler.transform(arr.reshape(-1, sh[-1])).reshape(sh)
    
    X_tr_s, X_va_s, X_te_s = scale(X_train), scale(X_val), scale(X_test)
    
    print(f"Starting parallel training on {len(MODEL_NAMES)} models...")
    summary_rows, detailed_rows = [], []
    all_preds = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(MODEL_NAMES)) as executor:
        futures = {executor.submit(train_single_model_task, name, X_tr_s, y_train, X_va_s, y_val, X_te_s, y_test, crops_test, len(selected)): name for name in MODEL_NAMES}
        for future in concurrent.futures.as_completed(futures):
            res_summary, res_detailed, y_pred = future.result()
            summary_rows.append(res_summary)
            detailed_rows.extend(res_detailed)
            all_preds[futures[future]] = y_pred

    pd.DataFrame(detailed_rows).to_csv(BASE / 'detailed_metrics.csv', index=False)
    pd.DataFrame(summary_rows).to_csv(BASE / 'model_comparison.csv', index=False)

    total_min = round((time.time() - t_start) / 60, 1)
    print('='*68)
    print(f"Total training: {total_min} minutes")
    print('='*68)

if __name__ == '__main__':
    main()
