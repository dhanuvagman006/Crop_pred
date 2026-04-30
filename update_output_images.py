# ============================================================
#  update_output_images.py
#  Regenerate plot images in outputs/ from saved metrics,
#  histories, and predictions.
# ============================================================

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path('outputs')
HISTORY_DIR = OUTPUT_DIR / 'histories'
PRED_DIR = OUTPUT_DIR / 'predictions'

MODEL_COLORS = {
    'LSTM': '#2196F3',
    'BiLSTM': '#4CAF50',
    'GRU': '#FF9800',
    'CNN-LSTM': '#9C27B0',
    'Transformer': '#F44336',
    'TCN': '#00BCD4',
}

CROP_ORDER = ['Rice', 'Coconut', 'Arecanut', 'Banana', 'Black pepper', 'Cocoa', 'Cashewnut', 'Mango']
MODEL_ORDER = ['LSTM', 'BiLSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'TCN']


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] Missing {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] Could not read {path}: {exc}")
        return None


def regenerate_metric_comparison(model_comp: pd.DataFrame) -> None:
    df = model_comp.copy()
    df = df.sort_values('Composite', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df['Model'], df['Avg_R2'], color=[MODEL_COLORS.get(m, '#3b82f6') for m in df['Model']])
    axes[0].set_title('Average R² by Model')
    axes[0].set_ylabel('R²')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(df['Model'], df['Avg_MAPE'], color=[MODEL_COLORS.get(m, '#3b82f6') for m in df['Model']])
    axes[1].set_title('Average MAPE by Model')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].grid(axis='y', alpha=0.25)

    _save_fig(fig, OUTPUT_DIR / 'metric_comparison_bar.png')


def regenerate_efficiency(model_comp: pd.DataFrame) -> None:
    df = model_comp.copy().sort_values('Total_Train_Time', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df['Model'], df['Total_Train_Time'], color=[MODEL_COLORS.get(m, '#3b82f6') for m in df['Model']])
    axes[0].set_title('Training Time by Model')
    axes[0].set_ylabel('Seconds')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(df['Model'], df['Params'] / 1000.0, color=[MODEL_COLORS.get(m, '#3b82f6') for m in df['Model']])
    axes[1].set_title('Model Size by Model')
    axes[1].set_ylabel('Parameters (K)')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].grid(axis='y', alpha=0.25)

    _save_fig(fig, OUTPUT_DIR / 'efficiency.png')


def regenerate_r2_heatmap(detailed: pd.DataFrame) -> None:
    pivot = detailed.pivot_table(index='Crop', columns='Model', values='R2', aggfunc='mean')
    pivot = pivot.reindex(index=[c for c in CROP_ORDER if c in pivot.index], columns=[m for m in MODEL_ORDER if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'R²'})
    ax.set_title('R² Heatmap by Crop and Model')
    _save_fig(fig, OUTPUT_DIR / 'r2_heatmap.png')


def regenerate_all_models_all_crops(detailed: pd.DataFrame) -> None:
    pivot = detailed.pivot_table(index='Crop', columns='Model', values='R2', aggfunc='mean')
    pivot = pivot.reindex(index=[c for c in CROP_ORDER if c in pivot.index], columns=[m for m in MODEL_ORDER if m in pivot.columns])

    n_crops = len(pivot.index)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axes = axes.flatten()

    for idx, crop in enumerate(pivot.index):
        ax = axes[idx]
        series = pivot.loc[crop].dropna()
        ax.bar(series.index, series.values, color=[MODEL_COLORS.get(m, '#3b82f6') for m in series.index])
        ax.set_title(crop)
        ax.set_ylim(min(-1.0, float(np.nanmin(pivot.values))), max(1.0, float(np.nanmax(pivot.values))))
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=0.2)

    for idx in range(n_crops, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Model R² Across All Crops', y=1.02, fontsize=14)
    _save_fig(fig, OUTPUT_DIR / 'all_models_all_crops.png')


def regenerate_actual_vs_pred(detailed: pd.DataFrame) -> None:
    best_by_crop = detailed.sort_values(['Crop', 'R2'], ascending=[True, False]).groupby('Crop', as_index=False).first()

    for _, row in best_by_crop.iterrows():
        crop = row['Crop']
        model = row['Model']
        pred_path = PRED_DIR / f"{model}_{crop.replace(' ', '_')}_predictions.csv"
        pred_df = _read_csv(pred_path)

        fig, ax = plt.subplots(figsize=(6, 6))
        if pred_df is None or pred_df.empty:
            ax.text(0.5, 0.5, f'No prediction data for {crop}\nRun training to populate {pred_path.name}',
                    ha='center', va='center', fontsize=11)
            ax.set_axis_off()
        else:
            ax.scatter(pred_df['Actual'], pred_df['Predicted'], alpha=0.7, color=MODEL_COLORS.get(model, '#3b82f6'))
            mn = float(np.nanmin([pred_df['Actual'].min(), pred_df['Predicted'].min()]))
            mx = float(np.nanmax([pred_df['Actual'].max(), pred_df['Predicted'].max()]))
            ax.plot([mn, mx], [mn, mx], '--', color='black', linewidth=1)
            ax.set_xlabel('Actual Yield')
            ax.set_ylabel('Predicted Yield')
            ax.set_title(f'{crop} - {model}')
            ax.grid(alpha=0.25)

        _save_fig(fig, OUTPUT_DIR / f'actual_vs_pred_{crop.replace(" ", "_")}.png')


def regenerate_loss_curves(detailed: pd.DataFrame) -> None:
    best_by_crop = detailed.sort_values(['Crop', 'R2'], ascending=[True, False]).groupby('Crop', as_index=False).first()
    n = len(best_by_crop)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (_, row) in enumerate(best_by_crop.iterrows()):
        crop = row['Crop']
        model = row['Model']
        hist_path = HISTORY_DIR / f"{model}_{crop.replace(' ', '_')}_history.csv"
        hist_df = _read_csv(hist_path)
        ax = axes[idx]

        if hist_df is None or hist_df.empty:
            ax.text(0.5, 0.5, f'No history for {crop}\nRun training to populate {hist_path.name}',
                    ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            continue

        ax.plot(hist_df['epoch'], hist_df['loss'], label='train loss', linewidth=2)
        if 'val_loss' in hist_df:
            ax.plot(hist_df['epoch'], hist_df['val_loss'], label='val loss', linewidth=2)
        ax.set_title(f'{crop} - {model}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Training Loss Curves for Best Model per Crop', y=1.01, fontsize=14)
    _save_fig(fig, OUTPUT_DIR / 'loss_curves.png')


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate output images from saved metrics and predictions')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PNGs in outputs/')
    args = parser.parse_args()

    _ensure_dir(OUTPUT_DIR)
    _ensure_dir(HISTORY_DIR)
    _ensure_dir(PRED_DIR)

    model_comp = _read_csv(OUTPUT_DIR / 'model_comparison.csv')
    detailed = _read_csv(OUTPUT_DIR / 'detailed_metrics.csv')

    if model_comp is None or detailed is None:
        raise SystemExit('Missing outputs/model_comparison.csv or outputs/detailed_metrics.csv')

    regenerate_metric_comparison(model_comp)
    regenerate_efficiency(model_comp)
    regenerate_r2_heatmap(detailed)
    regenerate_all_models_all_crops(detailed)
    regenerate_actual_vs_pred(detailed)
    regenerate_loss_curves(detailed)

    print('Updated plot images in outputs/')


if __name__ == '__main__':
    main()
