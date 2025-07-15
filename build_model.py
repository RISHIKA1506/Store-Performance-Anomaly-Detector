# build_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("--- Starting: Store Performance Analysis ---")

# ================================================================
#               1. LOAD AND MERGE DATA
# ================================================================

print("Loading data files...")
try:
    train_df = pd.read_csv(r'C:\MEEE\Store Performance Anomaly Detector\DATA\train.csv')
    stores_df = pd.read_csv(r'C:\MEEE\Store Performance Anomaly Detector\DATA\stores.csv')
    features_df = pd.read_csv(r'C:\MEEE\Store Performance Anomaly Detector\DATA\features.csv')
except FileNotFoundError as e:
    print(f"ERROR: Could not find a data file. Make sure train.csv, stores.csv, and features.csv are in the same folder.")
    print(e)
    exit()

print("Merging data files...")
# Merge the datasets into one master DataFrame
df = pd.merge(train_df, stores_df, how='left', on='Store')
df = pd.merge(df, features_df, how='left', on=['Store', 'Date', 'IsHoliday'])

# ================================================================
#               2. CLEAN AND PREPARE DATA
# ================================================================

print("Cleaning and preparing data...")

# Convert Date column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Fill missing values for MarkDown columns with 0 (assuming NaN means no promotion)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    df[col] = df[col].fillna(0)

# Fill missing CPI and Unemployment with the median value of their respective store
df['CPI'] = df.groupby('Store')['CPI'].transform(lambda x: x.fillna(x.median()))
df['Unemployment'] = df.groupby('Store')['Unemployment'].transform(lambda x: x.fillna(x.median()))

# Check if any NaNs are left
if df.isnull().sum().any():
    print("Warning: Missing values still exist after cleaning.")
    # For this project, we'll just drop any remaining rare NaNs
    df.dropna(inplace=True)


print("Data loaded, merged, and cleaned successfully.")
print("Master DataFrame shape:", df.shape)
print(df.head())
# build_model.py (continued...)

# ================================================================
#               3. FEATURE ENGINEERING
# ================================================================
print("\nPerforming feature engineering...")

# Create time-based features
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Year'] = df['Date'].dt.year

# Convert 'Type' column into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Type'], prefix='StoreType')

# Convert boolean 'IsHoliday' to integer (1 for True, 0 for False)
df['IsHoliday'] = df['IsHoliday'].astype(int)

print("Feature engineering complete.")
print("DataFrame columns after engineering:", df.columns.tolist())


# ================================================================
#           4. BUILD THE "EXPECTED SALES" MODEL (XGBoost)
# ================================================================
print("\nBuilding the sales forecasting model...")

# Define our features (X) and target (y)
# We drop the original Date and the target variable itself
features = [col for col in df.columns if col not in ['Date', 'Weekly_Sales']]
target = 'Weekly_Sales'

X = df[features]
y = df[target]

# Split data into training and testing sets
# We use a simple 80/20 split for this project
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# Initialize and train the XGBoost Regressor model
# We use some standard parameters. n_estimators is the number of trees.
# A lower learning_rate makes the model learn slower but often better.
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,          # A smaller number for faster training
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1,                 # Use all available CPU cores
    random_state=42
)

print("Training XGBoost model... (This may take a minute or two)")
model.fit(X_train, y_train)

print("Model training complete.")

# Evaluate the model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model evaluation complete. Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# Save the trained model to a file for later use
import joblib
joblib.dump(model, 'sales_forecasting_model.joblib')
print("Forecasting model saved to 'sales_forecasting_model.joblib'")
# build_model.py (continued...)

# ================================================================
#       5. ANOMALY DETECTION & ROOT CAUSE ANALYSIS INSIGHTS
# ================================================================
print("\n--- Starting Anomaly Detection & Analysis ---")

# Use our trained model to predict sales for the ENTIRE dataset
# This gives us the "Expected Sales" for every single week
print("Generating 'Expected_Sales' for the entire dataset...")
df['Expected_Sales'] = model.predict(X) # X is our full feature set from earlier

# Calculate the Performance Factor. Values < 1.0 are underperforming.
# We add a small number to avoid division by zero if expected sales are 0.
df['Performance_Factor'] = df['Weekly_Sales'] / (df['Expected_Sales'] + 0.01)

# We are only interested in cases where stores sold less than expected
underperforming_df = df[df['Performance_Factor'] < 0.9].copy()

# Sort to find the absolute worst-performing store-weeks
# These are our top anomalies to investigate
worst_performers = underperforming_df.sort_values(by='Performance_Factor').head(10)

print("\nTop 5 Worst-Performing Store-Weeks (Anomalies):")
# Display the top 5 anomalies in a readable format
print(worst_performers[[
    'Store', 'Dept', 'Date', 'Weekly_Sales', 'Expected_Sales', 'Performance_Factor'
]].head())


# --- Simple Root Cause Analysis for the #1 Worst Performer ---
print("\n--- Automated Root Cause Analysis ---")

# Get the single worst performing row from our data
top_anomaly = worst_performers.iloc[0]

print(f"Analyzing top anomaly: Store {top_anomaly['Store']}, Dept {top_anomaly['Dept']} on {top_anomaly['Date'].date()}")
print(f"Performance Factor: {top_anomaly['Performance_Factor']:.2f} (Sold only {top_anomaly['Performance_Factor']:.0%} of expected sales)")

# Calculate the average values for "normal" weeks for this specific store and department
normal_conditions = df[
    (df['Store'] == top_anomaly['Store']) &
    (df['Dept'] == top_anomaly['Dept'])
]

# Compare the anomaly's features to the normal average
print("\nPotential Contributing Factors:")
key_factors = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
for factor in key_factors:
    normal_avg = normal_conditions[factor].mean()
    anomaly_val = top_anomaly[factor]
    
    # Check if the anomaly's value is significantly different from the average
    if abs(anomaly_val - normal_avg) > normal_avg * 0.10: # If it's more than a 10% deviation
        print(f"- {factor} was {anomaly_val:.2f} (compared to a normal average of {normal_avg:.2f})")

# Check for lack of promotions
markdown_total_anomaly = top_anomaly[markdown_cols].sum()
if markdown_total_anomaly == 0:
    print("- No markdown promotions were active this week.")

# Final save of the enriched dataframe for a separate analysis script
df.to_csv('analysis_data.csv', index=False)
print("\nFull analysis data saved to 'analysis_data.csv'")

print("\n--- Build process complete! ---")