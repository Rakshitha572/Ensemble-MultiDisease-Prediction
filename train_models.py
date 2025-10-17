import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore

# -----------------------------
# Step 1: Load all datasets
# -----------------------------
diabetes = pd.read_csv('data/diabetes.csv')
heart = pd.read_csv('data/heart.csv')
kidney = pd.read_csv('data/kidney.csv')
parkinsons = pd.read_csv('data/parkinsons.csv')

# -----------------------------
# Step 2: Standardize column names
# -----------------------------
# Diabetes
diabetes.rename(columns={
    'BloodPressure': 'blood_pressure',
    'Glucose': 'glucose',
    'BMI': 'bmi',
    'Age': 'age',
    'Insulin': 'insulin',
    'Outcome': 'target'
}, inplace=True)

# Heart
heart.rename(columns={
    'age': 'age',
    'sex': 'gender',
    'trestbps': 'blood_pressure',
    'chol': 'cholesterol',
    'thalach': 'heart_rate',
    'target': 'target',
    'condition': 'target'  # in case CSV uses 'condition'
}, inplace=True)

# Kidney
kidney.rename(columns={
    'age': 'age',
    'bp': 'blood_pressure',
    'sg': 'specific_gravity',
    'al': 'albumin',
    'su': 'sugar',
    'rbc': 'rbc',
    'pc': 'pus_cell',
    'pcc': 'pus_cell_clumps',
    'ba': 'bacteria',
    'bgr': 'blood_glucose_random',
    'bu': 'blood_urea',
    'sc': 'creatinine',
    'sod': 'sodium',
    'pot': 'potassium',
    'hemo': 'hemoglobin',
    'pcv': 'packed_cell_volume',
    'wbcc': 'white_blood_cell_count',
    'rc': 'red_blood_cell_count',
    'htn': 'hypertension',
    'dm': 'diabetes_mellitus',
    'cad': 'coronary_artery_disease',
    'appet': 'appetite',
    'pe': 'pedal_edema',
    'ane': 'anemia',
    'classification': 'target'
}, inplace=True)

# Parkinson's
parkinsons.rename(columns={
    'name': 'patient_id',
    'status': 'target'
}, inplace=True)

# -----------------------------
# Step 3: Clean and Handle Missing Values
# -----------------------------
def fill_missing(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    return df_clean

# Strip whitespace from kidney categorical columns
for col in ['diabetes_mellitus', 'coronary_artery_disease', 'target']:
    if col in kidney.columns:
        kidney[col] = kidney[col].astype(str).str.strip()

# Convert numeric-like kidney columns to float safely
numeric_cols_kidney = [
    'age', 'blood_pressure', 'blood_urea', 'creatinine',
    'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count'
]
numeric_cols_kidney = [col for col in numeric_cols_kidney if col in kidney.columns]

for col in numeric_cols_kidney:
    kidney[col] = pd.to_numeric(kidney[col], errors='coerce')

# Fill missing values
diabetes = fill_missing(diabetes)
heart = fill_missing(heart)
kidney = fill_missing(kidney)
parkinsons = fill_missing(parkinsons)

# -----------------------------
# Step 4: Encode categorical features
# -----------------------------
le = LabelEncoder()

# Heart
if 'gender' in heart.columns:
    heart['gender'] = le.fit_transform(heart['gender'].astype(str))

# Kidney categorical columns
categorical_cols = [
    'rbc', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
    'appetite', 'pedal_edema', 'anemia', 'target'
]
for col in categorical_cols:
    if col in kidney.columns:
        kidney[col] = le.fit_transform(kidney[col].astype(str))

# -----------------------------
# Step 5: Normalize numeric features
# -----------------------------
scaler = StandardScaler()

# Diabetes
numeric_cols_diabetes = ['glucose', 'blood_pressure', 'bmi', 'insulin', 'age']
diabetes[numeric_cols_diabetes] = scaler.fit_transform(diabetes[numeric_cols_diabetes])

# Heart
numeric_cols_heart = ['age', 'blood_pressure', 'cholesterol', 'heart_rate']
heart[numeric_cols_heart] = scaler.fit_transform(heart[numeric_cols_heart])

# Kidney
kidney[numeric_cols_kidney] = scaler.fit_transform(kidney[numeric_cols_kidney])

# Parkinson's: all numeric columns except patient_id and target
numeric_cols_parkinsons = parkinsons.select_dtypes(include=['float64', 'int64']).columns
parkinsons[numeric_cols_parkinsons] = scaler.fit_transform(parkinsons[numeric_cols_parkinsons])

# -----------------------------
# Step 6: Save cleaned datasets
# -----------------------------
diabetes.to_csv('data/cleaned/diabetes_clean.csv', index=False)
heart.to_csv('data/cleaned/heart_clean.csv', index=False)
kidney.to_csv('data/cleaned/kidney_clean.csv', index=False)
parkinsons.to_csv('data/cleaned/parkinsons_clean.csv', index=False)

print("✅ All datasets loaded, cleaned, and saved successfully!")
