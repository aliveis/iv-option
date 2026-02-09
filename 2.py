import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, silhouette_score
import os

print("=" * 80)
print("MACHINE LEARNING PROJECT - AAPL OPTIONS ANALYSIS")
print("Dataset: AAPL Option Chains 2016-2023")
print("=" * 80)

# ============================================================================
# [1] LOADING DATASETS (TWO SEPARATE FILES)
# ============================================================================
print("\n[STEP 1] Loading Datasets...")

# فایل اول: 2016-2020
dataset_path_1 = 'datasets/AAPL_options.csv'
# فایل دوم: 2021-2023 ⬇️ نام فایل دوم را اینجا بنویسید
dataset_path_2 = 'datasets/AAPL_options_2.csv'

if not os.path.exists(dataset_path_1):
    print(f"\n⚠ ERROR: File not found: {dataset_path_1}")
    exit(1)

if not os.path.exists(dataset_path_2):
    print(f"\n⚠ ERROR: File not found: {dataset_path_2}")
    print("\nPlease update 'dataset_path_2' with the correct filename.")
    exit(1)

# بارگذاری دو فایل
print(f"  Loading file 1 (2016-2020): {dataset_path_1}")
data_1 = pd.read_csv(dataset_path_1)
print(f"  ✓ Loaded: {data_1.shape[0]:,} rows × {data_1.shape[1]} columns")

print(f"  Loading file 2 (2021-2023): {dataset_path_2}")
data_2 = pd.read_csv(dataset_path_2)
print(f"  ✓ Loaded: {data_2.shape[0]:,} rows × {data_2.shape[1]} columns")

# ترکیب دو دیتاست
print("\n  Merging datasets...")
data = pd.concat([data_1, data_2], ignore_index=True)
print(f"✓ Combined dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")

# ============================================================================
# [2] DATA PREPROCESSING & TRANSFORMATION
# ============================================================================
print("\n[STEP 2] Data Preprocessing & Transformation...")

# Convert date columns
data['Date'] = pd.to_datetime(data[' [QUOTE_DATE]'])
data['Year'] = data['Date'].dt.year

print(f"  Year range: {data['Year'].min()} - {data['Year'].max()}")
print(f"  Years available: {sorted(data['Year'].unique())}")

# Clean column names (remove spaces and brackets)
data.columns = data.columns.str.strip()


# Function to safely convert to numeric
def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')


# Process Call options
print("\n  Processing Call Options...")
calls = data[[
    'Date', 'Year', '[UNDERLYING_LAST]', '[STRIKE]', '[DTE]',
    '[C_LAST]', '[C_BID]', '[C_ASK]', '[C_VOLUME]', '[C_IV]',
    '[C_DELTA]', '[C_GAMMA]', '[C_VEGA]', '[C_THETA]'
]].copy()

calls.columns = ['Date', 'Year', 'UnderlyingPrice', 'StrikePrice', 'DTE',
                 'Last', 'Bid', 'Ask', 'Volume', 'IV',
                 'Delta', 'Gamma', 'Vega', 'Theta']
calls['Type'] = 'Call'

# Process Put options
print("  Processing Put Options...")
puts = data[[
    'Date', 'Year', '[UNDERLYING_LAST]', '[STRIKE]', '[DTE]',
    '[P_LAST]', '[P_BID]', '[P_ASK]', '[P_VOLUME]', '[P_IV]',
    '[P_DELTA]', '[P_GAMMA]', '[P_VEGA]', '[P_THETA]'
]].copy()

puts.columns = ['Date', 'Year', 'UnderlyingPrice', 'StrikePrice', 'DTE',
                'Last', 'Bid', 'Ask', 'Volume', 'IV',
                'Delta', 'Gamma', 'Vega', 'Theta']
puts['Type'] = 'Put'

# Combine both
combined = pd.concat([calls, puts], ignore_index=True)
print(f"✓ Combined dataset: {len(combined):,} rows (Calls: {len(calls):,}, Puts: {len(puts):,})")

# Convert numeric columns
numeric_cols = ['UnderlyingPrice', 'StrikePrice', 'DTE', 'Last', 'Bid', 'Ask',
                'Volume', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta']
for col in numeric_cols:
    combined[col] = safe_numeric(combined[col])

# Remove rows with missing critical values
combined = combined.dropna(subset=['IV', 'Last', 'Volume', 'StrikePrice', 'Date',
                                   'UnderlyingPrice', 'Bid', 'Ask', 'DTE'])
combined = combined[combined['IV'] > 0]  # Remove zero/negative IV
combined = combined[combined['Volume'] > 0]  # Only liquid options
combined = combined[combined['UnderlyingPrice'] > 0]  # Valid underlying price
print(f"✓ After cleaning: {len(combined):,} rows")

# Feature Engineering
print("\n  Engineering Features...")
combined['Moneyness'] = combined['StrikePrice'] / combined['UnderlyingPrice']
combined['Volume_log'] = np.log1p(combined['Volume'])
combined['Bid_Ask_Spread'] = combined['Ask'] - combined['Bid']
combined['Mid_Price'] = (combined['Bid'] + combined['Ask']) / 2

# Sort by Type and Date for proper next-day IV calculation
combined = combined.sort_values(['Type', 'Date', 'StrikePrice']).reset_index(drop=True)

# Create next-day IV target (grouped by option type)
combined['IV_next'] = combined.groupby('Type')['IV'].shift(-1)

# **CRITICAL FIX**: Remove any remaining NaN values after feature engineering
print(f"  Before final cleaning: {len(combined):,} rows")
feature_cols = ['StrikePrice', 'Last', 'Bid', 'Ask', 'Volume_log',
                'Moneyness', 'DTE', 'IV', 'IV_next']
combined = combined.dropna(subset=feature_cols)
print(f"✓ After final NaN removal: {len(combined):,} rows")

# Verify no NaN values remain
nan_counts = combined[feature_cols].isna().sum()
if nan_counts.sum() > 0:
    print("\n⚠ WARNING: NaN values still present:")
    print(nan_counts[nan_counts > 0])
else:
    print("  ✓ No NaN values detected in feature columns")

# Time Series Split: 2016-2020 Train, 2021-2023 Test
train_data = combined[combined['Year'] <= 2020].copy()
test_data = combined[combined['Year'] >= 2021].copy()

print(f"\n  ✅ Time Series Split:")
print(f"    Train (2016-2020): {len(train_data):,} rows ({train_data['Year'].min()}-{train_data['Year'].max()})")
print(f"    Test (2021-2023):  {len(test_data):,} rows ({test_data['Year'].min()}-{test_data['Year'].max()})")

if len(train_data) == 0 or len(test_data) == 0:
    print("\n⚠ ERROR: Empty train or test set!")
    print(f"   Years in dataset: {sorted(combined['Year'].unique())}")
    exit(1)

# ============================================================================
# [3] PART 1: REGRESSION - PREDICTING NEXT DAY IV
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: REGRESSION - PREDICTING NEXT DAY IMPLIED VOLATILITY")
print("=" * 80)

feature_cols_reg = ['StrikePrice', 'Last', 'Bid', 'Ask', 'Volume_log',
                    'Moneyness', 'DTE', 'IV']

X_reg_train = train_data[feature_cols_reg]
y_reg_train = train_data['IV_next']
X_reg_test = test_data[feature_cols_reg]
y_reg_test = test_data['IV_next']

print(f"\n  Features: {feature_cols_reg}")
print(f"  Train samples: {len(X_reg_train):,}")
print(f"  Test samples:  {len(X_reg_test):,}")

# Double-check for NaN values before scaling
print(f"\n  Checking for NaN values before training...")
train_nans = X_reg_train.isna().sum().sum()
test_nans = X_reg_test.isna().sum().sum()
print(f"    Train NaNs: {train_nans}")
print(f"    Test NaNs:  {test_nans}")

if train_nans > 0 or test_nans > 0:
    print("\n  ⚠ Removing remaining NaN rows...")
    X_reg_train = X_reg_train.dropna()
    y_reg_train = y_reg_train[X_reg_train.index]
    X_reg_test = X_reg_test.dropna()
    y_reg_test = y_reg_test[X_reg_test.index]
    print(f"  ✓ After cleanup - Train: {len(X_reg_train):,}, Test: {len(X_reg_test):,}")

# Scaling
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Define models
models_reg = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=7),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                   max_iter=200, early_stopping=True, validation_fraction=0.15,
                                   random_state=42, verbose=False)
}

print("\n  Training Regression Models...")
results_reg = []
for name, model in models_reg.items():
    print(f"    → {name:20s}", end=" ", flush=True)
    model.fit(X_reg_train_scaled, y_reg_train)
    y_pred = model.predict(X_reg_test_scaled)

    mse = mean_squared_error(y_reg_test, y_pred)
    mae = np.mean(np.abs(y_reg_test - y_pred))
    r2 = r2_score(y_reg_test, y_pred)

    results_reg.append({'Model': name, 'MSE': mse, 'MAE': mae, 'R²': r2})
    print(f"✓ MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

results_reg_df = pd.DataFrame(results_reg)

# Visualize Regression Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].barh(results_reg_df['Model'], results_reg_df['MSE'], color='salmon')
axes[0].set_xlabel('Mean Squared Error', fontsize=12)
axes[0].set_title('Regression: MSE Comparison', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()

axes[1].barh(results_reg_df['Model'], results_reg_df['MAE'], color='lightcoral')
axes[1].set_xlabel('Mean Absolute Error', fontsize=12)
axes[1].set_title('Regression: MAE Comparison', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()

axes[2].barh(results_reg_df['Model'], results_reg_df['R²'], color='skyblue')
axes[2].set_xlabel('R² Score', fontsize=12)
axes[2].set_title('Regression: R² Score', fontsize=13, fontweight='bold')
axes[2].invert_yaxis()
axes[2].axvline(x=0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Plot saved: regression_results.png")
plt.show()

# Feature Importance (Random Forest)
print("\n  Feature Importance (Random Forest):")
rf_model = models_reg['Random Forest']
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_cols_reg,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for idx, row in feature_imp_df.iterrows():
    print(f"    {row['Feature']:20s}: {row['Importance']:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='teal')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest - Feature Importance (Regression)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_regression.png', dpi=150, bbox_inches='tight')
print("  ✓ Plot saved: feature_importance_regression.png")
plt.show()

# ============================================================================
# [4] PART 2: CLASSIFICATION - PREDICTING IV DIRECTION (±2% THRESHOLD)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CLASSIFICATION - PREDICTING IV DIRECTION (±2% THRESHOLD)")
print("=" * 80)

# Calculate IV percentage change
train_data['IV_change_pct'] = ((train_data['IV_next'] - train_data['IV']) / train_data['IV']) * 100
test_data['IV_change_pct'] = ((test_data['IV_next'] - test_data['IV']) / test_data['IV']) * 100

# Create labels: 1 = Up (>2%), 0 = Down (<-2%), 2 = Neutral (filtered out)
train_data['Label'] = train_data['IV_change_pct'].apply(
    lambda x: 1 if x > 2 else (0 if x < -2 else 2)
)
test_data['Label'] = test_data['IV_change_pct'].apply(
    lambda x: 1 if x > 2 else (0 if x < -2 else 2)
)

# Filter out neutral cases
train_class = train_data[train_data['Label'] != 2].copy()
test_class = test_data[test_data['Label'] != 2].copy()

print(f"\n  Classification Threshold: ±2%")
print(
    f"    Train: {len(train_class):,} samples (Up: {sum(train_class['Label'] == 1):,}, Down: {sum(train_class['Label'] == 0):,})")
print(
    f"    Test:  {len(test_class):,} samples (Up: {sum(test_class['Label'] == 1):,}, Down: {sum(test_class['Label'] == 0):,})")

X_class_train = train_class[feature_cols_reg]
y_class_train = train_class['Label']
X_class_test = test_class[feature_cols_reg]
y_class_test = test_class['Label']

# Scaling
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

# Define models
models_class = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                    max_iter=200, early_stopping=True, validation_fraction=0.15,
                                    random_state=42, verbose=False)
}

print("\n  Training Classification Models...")
results_class = []
for name, model in models_class.items():
    print(f"    → {name:20s}", end=" ", flush=True)
    model.fit(X_class_train_scaled, y_class_train)
    y_pred = model.predict(X_class_test_scaled)

    # Metrics
    acc = accuracy_score(y_class_test, y_pred)
    prec = precision_score(y_class_test, y_pred, zero_division=0)
    rec = recall_score(y_class_test, y_pred, zero_division=0)
    f1 = f1_score(y_class_test, y_pred, zero_division=0)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_class_test_scaled)[:, 1]
        auc = roc_auc_score(y_class_test, y_proba)
    else:
        auc = None

    results_class.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'AUC': auc
    })
    auc_str = f"{auc:.4f}" if auc else "N/A"
    print(f"✓ Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_str}")

results_class_df = pd.DataFrame(results_class)

# Visualize Classification Results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].barh(results_class_df['Model'], results_class_df['Accuracy'], color='lightgreen')
axes[0, 0].set_xlabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Classification: Accuracy', fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()

axes[0, 1].barh(results_class_df['Model'], results_class_df['Precision'], color='gold')
axes[0, 1].set_xlabel('Precision', fontsize=12)
axes[0, 1].set_title('Classification: Precision', fontsize=13, fontweight='bold')
axes[0, 1].invert_yaxis()

axes[1, 0].barh(results_class_df['Model'], results_class_df['Recall'], color='orange')
axes[1, 0].set_xlabel('Recall', fontsize=12)
axes[1, 0].set_title('Classification: Recall', fontsize=13, fontweight='bold')
axes[1, 0].invert_yaxis()

axes[1, 1].barh(results_class_df['Model'], results_class_df['F1'], color='coral')
axes[1, 1].set_xlabel('F1 Score', fontsize=12)
axes[1, 1].set_title('Classification: F1 Score', fontsize=13, fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Plot saved: classification_results.png")
plt.show()

# Feature Importance (Random Forest)
print("\n  Feature Importance (Random Forest - Classification):")
rf_class_model = models_class['Random Forest']
importances_class = rf_class_model.feature_importances_
feature_imp_class_df = pd.DataFrame({
    'Feature': feature_cols_reg,
    'Importance': importances_class
}).sort_values('Importance', ascending=False)

for idx, row in feature_imp_class_df.iterrows():
    print(f"    {row['Feature']:20s}: {row['Importance']:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_class_df['Feature'], feature_imp_class_df['Importance'], color='purple')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest - Feature Importance (Classification)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_classification.png', dpi=150, bbox_inches='tight')
print("  ✓ Plot saved: feature_importance_classification.png")
plt.show()

# ============================================================================
# [5] PART 3: CLUSTERING - OPTION SEGMENTATION (K-MEANS, K=3)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: CLUSTERING - OPTION SEGMENTATION (K-MEANS, K=3)")
print("=" * 80)

X_cluster = combined[feature_cols_reg].dropna()
print(f"\n  Clustering dataset: {len(X_cluster):,} samples")

# Scaling
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# K-Means clustering (K=3)
print("\n  Performing K-Means clustering (K=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)
sil_score = silhouette_score(X_cluster_scaled, clusters)

print(f"  ✓ Silhouette Score: {sil_score:.4f}")
print(f"\n  Cluster Distribution:")
unique, counts = np.unique(clusters, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"    Cluster {cluster_id}: {count:,} samples ({count / len(clusters) * 100:.1f}%)")

# Visualize Clustering
plt.figure(figsize=(14, 8))
scatter = plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1],
                      c=clusters, cmap='viridis', alpha=0.5, s=8, edgecolors='none')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=400, edgecolors='black', linewidths=2.5,
            label='Centroids', zorder=5)
plt.xlabel('Feature 1 (StrikePrice - Scaled)', fontsize=12)
plt.ylabel('Feature 2 (Last Price - Scaled)', fontsize=12)
plt.title(f'K-Means Clustering (K=3) | Silhouette Score: {sil_score:.4f}',
          fontsize=13, fontweight='bold')
plt.colorbar(scatter, label='Cluster ID')
plt.legend(fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('clustering_results.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Plot saved: clustering_results.png")
plt.show()

# Elbow Method
print("\n  Elbow Method Analysis (K=2 to K=10)...")
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
print("  ✓ Plot saved: elbow_method.png")
plt.show()

# ============================================================================
# [6] FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print("\n✓ PART 1 - REGRESSION (Next-Day IV Prediction):")
best_reg_model = results_reg_df.loc[results_reg_df['R²'].idxmax()]
print(f"    Best Model: {best_reg_model['Model']}")
print(f"    R² Score:   {best_reg_model['R²']:.4f}")
print(f"    MAE:        {best_reg_model['MAE']:.6f}")

print("\n✓ PART 2 - CLASSIFICATION (IV Direction ±2%):")
best_class_model = results_class_df.loc[results_class_df['F1'].idxmax()]
print(f"    Best Model: {best_class_model['Model']}")
print(f"    Accuracy:   {best_class_model['Accuracy']:.4f}")
print(f"    F1 Score:   {best_class_model['F1']:.4f}")

print("\n✓ PART 3 - CLUSTERING (K-Means, K=3):")
print(f"    Silhouette Score: {sil_score:.4f}")
print(f"    Total Samples:    {len(X_cluster):,}")

print("\n" + "=" * 80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("Generated Outputs:")
print("  • regression_results.png")
print("  • feature_importance_regression.png")
print("  • classification_results.png")
print("  • feature_importance_classification.png")
print("  • clustering_results.png")
print("  • elbow_method.png")
print("=" * 80)