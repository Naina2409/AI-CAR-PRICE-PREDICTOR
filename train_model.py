# train_model_force_fix.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚗 Car Price Model Training - FORCED CONVERSION")
print("="*60)

# Create folders
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load data - force all columns as string initially
print("\n📂 Loading data...")
df = pd.read_csv('data/car.csv', dtype=str)  # Force all as string
print(f"✅ Loaded {len(df)} rows")

print("\n📊 First 3 rows (raw):")
print(df.head(3))

# ============================================
# 1. CLEAN YEAR COLUMN - FORCE CONVERSION
# ============================================
print("\n📅 Cleaning Year column...")

def extract_year(val):
    if pd.isna(val):
        return 2015
    val_str = str(val)
    # Find all digits
    digits = re.findall(r'\d+', val_str)
    if digits:
        year = int(digits[0])
        if 1990 <= year <= 2024:
            return year
    return 2015

df['year_num'] = df['year'].apply(extract_year)
print(f"Original year values: {df['year'].head(5).tolist()}")
print(f"Converted year values: {df['year_num'].head(5).tolist()}")
df['year'] = df['year_num']
df = df.drop('year_num', axis=1)

# ============================================
# 2. CLEAN PRICE COLUMN
# ============================================
print("\n💰 Cleaning Price column...")

def extract_price(val):
    if pd.isna(val):
        return None
    val_str = str(val)
    # Remove commas, spaces, ₹
    cleaned = re.sub(r'[₹,\s]', '', val_str)
    # Extract digits
    digits = re.findall(r'\d+', cleaned)
    if digits:
        # Join digits (handles cases like "4,25,000" -> "425000")
        price_str = ''.join(digits)
        try:
            price = float(price_str)
            if 50000 <= price <= 5000000:
                return price
        except:
            pass
    return None

df['price_num'] = df['Price'].apply(extract_price)
print(f"Original Price: {df['Price'].head(5).tolist()}")
print(f"Converted Price: {df['price_num'].head(5).tolist()}")

# Remove invalid prices
df = df.dropna(subset=['price_num'])
df['Price'] = df['price_num']
df = df.drop('price_num', axis=1)
print(f"After price cleaning: {len(df)} rows")

# ============================================
# 3. CLEAN KMS COLUMN
# ============================================
print("\n📊 Cleaning kms_driven column...")

def extract_kms(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower()
    # Remove 'kms', 'km'
    val_str = re.sub(r'kms?', '', val_str)
    # Extract digits only
    digits = re.findall(r'\d+', val_str)
    if digits:
        kms = int(''.join(digits))
        return min(kms, 300000)  # Cap at 300k
    return 0

df['kms_num'] = df['kms_driven'].apply(extract_kms)
print(f"Original KMS: {df['kms_driven'].head(5).tolist()}")
print(f"Converted KMS: {df['kms_num'].head(5).tolist()}")
df['kms_driven'] = df['kms_num']
df = df.drop('kms_num', axis=1)

# ============================================
# 4. CREATE AGE COLUMN (Now year is numeric)
# ============================================
print("\n📅 Creating age column...")
# Verify year is numeric
print(f"Year column type: {df['year'].dtype}")
print(f"Sample years: {df['year'].head(5).tolist()}")

df['age'] = 2024 - df['year']
print(f"Sample ages: {df['age'].head(5).tolist()}")
df['age'] = df['age'].clip(0, 30)

# ============================================
# 5. REMOVE OUTLIERS
# ============================================
print("\n🧹 Removing outliers...")
initial = len(df)
df = df[df['Price'] > 50000]
df = df[df['Price'] < 5000000]
df = df[df['kms_driven'] < 300000]
df = df[df['year'] >= 1990]
print(f"Removed {initial - len(df)} outlier rows")

# ============================================
# 6. ENCODE CATEGORICAL VARIABLES
# ============================================
print("\n🏷️ Encoding categorical variables...")
label_encoders = {}

# Encode company
le_company = LabelEncoder()
df['company_encoded'] = le_company.fit_transform(df['company'].astype(str))
label_encoders['company'] = le_company
print(f"  ✓ Company: {len(le_company.classes_)} unique values")

# Encode name
le_name = LabelEncoder()
df['name_encoded'] = le_name.fit_transform(df['name'].astype(str))
label_encoders['name'] = le_name
print(f"  ✓ Name: {len(le_name.classes_)} unique values")

# Encode fuel_type
le_fuel = LabelEncoder()
df['fuel_type_encoded'] = le_fuel.fit_transform(df['fuel_type'].astype(str))
label_encoders['fuel_type'] = le_fuel
print(f"  ✓ Fuel Type: {len(le_fuel.classes_)} unique values")

# ============================================
# 7. PREPARE FEATURES
# ============================================
print("\n📊 Preparing features...")
feature_cols = ['company_encoded', 'name_encoded', 'fuel_type_encoded', 'year', 'kms_driven', 'age']
X = df[feature_cols].astype(float)  # Force all to float
y = df['Price'].astype(float)

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeature types:")
for col in feature_cols:
    print(f"  {col}: {X[col].dtype}")

# ============================================
# 8. TRAIN MODEL
# ============================================
print("\n🤖 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Calculate accuracy
score = model.score(X, y)
print(f"✅ Model R² Score: {score:.3f}")

# Feature importance
if hasattr(model, 'feature_importances_'):
    print("\n🎯 Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2%}")

# ============================================
# 9. SAVE FILES
# ============================================
print("\n💾 Saving files...")
joblib.dump(model, 'model/model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
df.to_csv('data/clean_data.csv', index=False)

print("\n" + "="*60)
print("✅ SUCCESS! Files created!")
print("="*60)

# Show final data info
print("\n📊 Final dataset summary:")
print(f"  Total cars: {len(df)}")
print(f"  Price range: ₹{df['Price'].min():,.0f} - ₹{df['Price'].max():,.0f}")
print(f"  Average price: ₹{df['Price'].mean():,.0f}")
print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
print(f"  Average kms: {df['kms_driven'].mean():,.0f}")
print(f"  Average age: {df['age'].mean():.1f} years")

# Test prediction
print("\n🎯 Sample predictions:")
for idx in range(min(3, len(df))):
    features = X.iloc[idx:idx+1]
    pred = model.predict(features)[0]
    actual = y.iloc[idx]
    error_pct = abs(actual - pred) / actual * 100
    print(f"\n  {idx+1}. {df['company'].iloc[idx]} {df['name'].iloc[idx][:30]}")
    print(f"     Year: {df['year'].iloc[idx]}, KMs: {df['kms_driven'].iloc[idx]:,}")
    print(f"     Actual: ₹{actual:,.0f} → Predicted: ₹{pred:,.0f} (Error: {error_pct:.1f}%)")

print("\n" + "="*60)
print("🚀 Training complete! Run:")
print("   python -m streamlit run app.py")
print("="*60)