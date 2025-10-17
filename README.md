Ensemble Multi-Disease Prediction System
📌 Overview

The Ensemble Multi-Disease Prediction System is a machine learning-based healthcare application that predicts the likelihood of multiple diseases — including Heart Disease, Diabetes, Kidney Disease, and Parkinson’s Disease.
It leverages ensemble learning techniques (Random Forest, XGBoost, Voting Classifier) to improve accuracy, generalization, and reliability across multiple medical datasets.

🎯 Objectives

Develop a unified framework to predict multiple diseases using machine learning.

Apply ensemble models for improved predictive accuracy.

Compare performance against individual baseline models.

Provide a user-friendly Flask web interface for real-time predictions.

Integrate visual analytics for explainability and feature insights.

Ensemble_MultiDisease_Prediction/
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── kidney.csv
│   ├── parkinsons.csv
│   └── cleaned/                # cleaned datasets
│
├── train_models.py             # dataset preprocessing
├── ensemble_training.py        # (to be added) ensemble model training
├── app.py                      # Flask web application
├── requirements.txt            # dependencies
├── venv/                       # virtual environment
└── README.md                   # project documentation


⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/<your-username>/Ensemble_MultiDisease_Prediction.git
cd Ensemble_MultiDisease_Prediction

2️⃣ Create a virtual environment
python -m venv venv
source venv/Scripts/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run preprocessing
python train_models.py

🧩 Technologies Used

Python 3.x

Scikit-learn

XGBoost

Pandas / NumPy

Matplotlib / Seaborn

Flask

SHAP / LIME (Explainable AI tools – upcoming)

🚀 Next Steps

 Train ensemble models for each disease.

 Integrate Explainable AI (XAI) using SHAP/LIME.

 Build Flask web dashboard for real-time predictions.

👩‍💻 Author

Developed by Rakshitha K
MCA Student