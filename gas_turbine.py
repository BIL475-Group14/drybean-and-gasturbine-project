import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# 1. SUPPRESS WARNINGS
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import catboost as cb


# 2. METRIC DEFINITION (SMAPE)
def calculate_smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)"""
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 3. DATA LOADING AND PREPROCESSING
# Ensure this path matches your file location
file_path = 'Gas_Turbine_Co_NoX_2015.mat'
if not os.path.exists(file_path):
    file_path = 'datasets/Gas_Turbine_Co_NoX_2015.mat'

data = scipy.io.loadmat(file_path) 
X_raw = data['feat']
y = data['lbl2'].flatten()

# --- DEFINE FEATURE NAMES ---
feature_names = [
    'AT',      # Ambient temperature (°C)
    'AP',      # Ambient pressure (mbar)
    'AH',      # Ambient humidity (%)
    'AFDP',    # Air filter difference pressure (mbar)
    'GTEP',    # Gas turbine exhaust pressure (mbar)
    'TIT',     # Turbine inlet temperature (°C)
    'TAT',     # Turbine after temperature (°C)
    'TEY',     # Turbine energy yield (MWH)
    'CDP',     # Compressor discharge pressure (mbar)
]

# Create DataFrame with names
df_X = pd.DataFrame(X_raw, columns=feature_names)

print(f"Dataset Loaded: {df_X.shape[0]} samples, {df_X.shape[1]} features.")

# --- 4. DISPLAY PLOT 1: FEATURE HISTOGRAMS ---
print("\n--- Displaying Feature Distributions ---")
plt.figure(figsize=(15, 10))
df_X.hist(bins=30, figsize=(15, 10), layout=(3, 3), color='teal', alpha=0.7, edgecolor='black')
plt.suptitle('Feature Distributions (Gas Turbine Data)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- PREPROCESSING (Scaling) ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df_X), columns=feature_names)

# --- 5. MODEL TRAINING (KNN & CatBoost) ---
kf = KFold(n_splits=3, shuffle=True, random_state=42)

models = {
    "KNN": KNeighborsRegressor(),
    "CatBoost": cb.CatBoostRegressor(verbose=0, random_state=42)
}

plot_data = {}
results_table = []

print("Training Models (3-Fold CV)...")

for name, model in models.items():
    y_true_all = []
    y_pred_all = []
    
    # K-Fold loop
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_p = model.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(np.array(y_p).flatten())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Metric Calculation
    mae_val = mean_absolute_error(y_true_all, y_pred_all)
    smape_val = calculate_smape(y_true_all, y_pred_all)
    
    results_table.append({"Model": name, "MAE": mae_val, "SMAPE (%)": smape_val})
    
    # Store data for plotting
    plot_data[name] = {
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'mae': mae_val,
        'smape': smape_val
    }

print("Training Completed.")

# --- 6. DISPLAY PLOT 2: SIDE-BY-SIDE COMPARISON ---
print("\n--- Displaying Prediction Accuracy ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Model Comparison: Actual vs Predicted (lbl2)', fontsize=16)

model_names = list(models.keys())

for i, ax in enumerate(axes):
    name = model_names[i]
    data = plot_data[name]
    
    # Downsample for clearer scatter plot if data is large (show 1000 points)
    y_t = data['y_true']
    y_p = data['y_pred']
    
    if len(y_t) > 1000:
        indices = np.random.choice(len(y_t), 1000, replace=False)
        y_t = y_t[indices]
        y_p = y_p[indices]

    # Scatter plot
    sns.scatterplot(x=y_t, y=y_p, ax=ax, alpha=0.5, color='royalblue')
    
    # Ideal Line
    line_min = min(y_t.min(), y_p.min())
    line_max = max(y_t.max(), y_p.max())
    ax.plot([line_min, line_max], [line_min, line_max], 'r--', lw=2, label='Ideal (x=y)')
    
    ax.set_title(f'{name} Regressor\nMAE: {data["mae"]:.4f} | SMAPE: {data["smape"]:.2f}%')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- 7. DISPLAY METRIC TABLE ---
df_results = pd.DataFrame(results_table).sort_values(by="MAE", ascending=True)

print("\n" + "="*45)
print("      REGRESSION PERFORMANCE RESULTS (lbl2)")
print("="*45)
print(df_results.to_string(index=False))
print("="*45)