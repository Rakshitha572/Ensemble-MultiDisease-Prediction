import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Create 'models' folder if not exists
os.makedirs("models", exist_ok=True)

def train_and_save_model(dataset_path, target_column, model_name):
    print(f"\n🔹 Training model for {model_name}...")

    df = pd.read_csv(dataset_path)

    # Strip whitespace from object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # For Parkinson's dataset, convert continuous target to int
    if model_name.lower() == 'parkinsons':
        df[target_column] = df[target_column].round().astype(int)

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale only numeric columns
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Define base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    lr = LogisticRegression(max_iter=500)

    # Ensemble model (soft voting)
    ensemble = VotingClassifier(estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lr', lr)
    ], voting='soft')

    # Train
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)

    # Evaluate
    print(f"✅ Accuracy for {model_name}: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    # Save model and scaler
    joblib.dump(ensemble, f"models/{model_name}_model.pkl")
    joblib.dump(scaler, f"models/{model_name}_scaler.pkl")
    print(f"💾 Saved: models/{model_name}_model.pkl\n")

# Train models for each disease
train_and_save_model("data/cleaned/diabetes_clean.csv", "target", "diabetes")
train_and_save_model("data/cleaned/heart_clean.csv", "target", "heart")
train_and_save_model("data/cleaned/kidney_clean.csv", "target", "kidney")
train_and_save_model("data/cleaned/parkinsons_clean.csv", "target", "parkinsons")

print("\n🎯 All ensemble models trained and saved successfully!")
