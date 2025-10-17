import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore

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
    'target': 'target'
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
    'bgr': 'blood_glucose_random',
    'bu': 'blood_urea',
    'sc': 'creatinine',
    'sod': 'sodium',
    'pot': 'potassium',
    'hemo': 'hemoglobin',
    'pcv': 'pcv',
    'wbcc': 'white_blood_cell_count',
    'rbcc': 'red_blood_cell_count',
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
# Step 3: Handle missing values
# -----------------------------
# Simple approach: fill numerical with mean, categorical with mode
def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    return df

diabetes = fill_missing(diabetes)
heart = fill_missing(heart)
kidney = fill_missing(kidney)
parkinsons = fill_missing(parkinsons)

# -----------------------------
# Step 4: Encode categorical features
# -----------------------------
le = LabelEncoder()

# Heart: gender
if 'gender' in heart.columns:
    heart['gender'] = le.fit_transform(heart['gender'])

# Kidney: categorical columns
categorical_cols = ['rbc','pc','htn','dm','cad','appet','pe','ane']
for col in categorical_cols:
    if col in kidney.columns:
        kidney[col] = le.fit_transform(kidney[col])

# -----------------------------
# Step 5: Normalize numeric features
# -----------------------------
scaler = StandardScaler()

numeric_cols_diabetes = ['glucose', 'blood_pressure', 'bmi', 'insulin', 'age']
diabetes[numeric_cols_diabetes] = scaler.fit_transform(diabetes[numeric_cols_diabetes])

numeric_cols_heart = ['age', 'blood_pressure', 'cholesterol', 'heart_rate']
heart[numeric_cols_heart] = scaler.fit_transform(heart[numeric_cols_heart])

numeric_cols_kidney = ['age', 'blood_pressure', 'blood_urea', 'creatinine', 'hemoglobin']
kidney[numeric_cols_kidney] = scaler.fit_transform(kidney[numeric_cols_kidney])

numeric_cols_parkinsons = parkinsons.columns.drop(['patient_id','target'])
parkinsons[numeric_cols_parkinsons] = scaler.fit_transform(parkinsons[numeric_cols_parkinsons])

# -----------------------------
# Step 6: Save cleaned datasets
# -----------------------------
diabetes.to_csv('data/cleaned/diabetes_clean.csv', index=False)
heart.to_csv('data/cleaned/heart_clean.csv', index=False)
kidney.to_csv('data/cleaned/kidney_clean.csv', index=False)
parkinsons.to_csv('data/cleaned/parkinsons_clean.csv', index=False)

print("✅ All datasets loaded, cleaned, and saved successfully!")
