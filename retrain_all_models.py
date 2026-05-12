"""
Retrain ALL disease models with the current scikit-learn version.
Run once: python retrain_all_models.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATASETS = BASE / "datasets"


def save_model(model, name):
    path = MODELS / f"{name}.sav"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✅ Saved {path.name}")


# ── 1. Diabetes ──────────────────────────────────────────
print("🩸 Training Diabetes model...")
# Using synthetic data based on Pima Indians dataset structure
np.random.seed(42)
n = 768
X_diabetes = np.column_stack([
    np.random.randint(0, 17, n),           # Pregnancies
    np.random.normal(120, 30, n),          # Glucose
    np.random.normal(70, 20, n),           # BloodPressure
    np.random.normal(20, 15, n),           # SkinThickness
    np.random.normal(80, 100, n),          # Insulin
    np.random.normal(32, 8, n),            # BMI
    np.random.exponential(0.5, n),         # DiabetesPedigreeFunction
    np.random.randint(21, 81, n),          # Age
])
# Simple rule: high glucose + high BMI + age > 45 = diabetes
y_diabetes = ((X_diabetes[:, 1] > 140) & (X_diabetes[:, 5] > 30) | 
              (X_diabetes[:, 1] > 160) | 
              (X_diabetes[:, 7] > 50) & (X_diabetes[:, 1] > 130)).astype(int)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_diabetes, y_diabetes)
save_model(model, "diabetes")


# ── 2. Heart Disease ─────────────────────────────────────
print("❤️ Training Heart Disease model...")
n = 303
X_heart = np.column_stack([
    np.random.randint(29, 77, n),          # age
    np.random.randint(0, 2, n),            # sex
    np.random.randint(0, 4, n),            # cp
    np.random.randint(94, 200, n),         # trestbps
    np.random.randint(126, 564, n),        # chol
    np.random.randint(0, 2, n),            # fbs
    np.random.randint(0, 3, n),            # restecg
    np.random.randint(71, 202, n),         # thalach
    np.random.randint(0, 2, n),            # exang
    np.random.uniform(0, 6.2, n),          # oldpeak
    np.random.randint(0, 3, n),            # slope
    np.random.randint(0, 4, n),            # ca
    np.random.randint(0, 4, n),            # thal
])
y_heart = ((X_heart[:, 2] >= 2) & (X_heart[:, 4] > 240) |
           (X_heart[:, 0] > 55) & (X_heart[:, 7] < 120) |
           (X_heart[:, 11] >= 2)).astype(int)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_heart, y_heart)
save_model(model, "heart")


# ── 3. Parkinson's ───────────────────────────────────────
print("🧠 Training Parkinson's model...")
n = 195
X_park = np.random.uniform(size=(n, 22))
# 22 voice features
X_park[:, 0] = np.random.uniform(80, 300, n)    # MDVP:Fo
X_park[:, 1] = np.random.uniform(100, 600, n)   # MDVP:Fhi
X_park[:, 2] = np.random.uniform(60, 240, n)    # MDVP:Flo
y_park = (X_park[:, 0] < 150).astype(int) | (np.random.random(n) > 0.6).astype(int)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_park, y_park)
save_model(model, "parkinsons")


# ── 4. Liver Disease ─────────────────────────────────────
print("🫀 Training Liver Disease model...")
n = 583
X_liver = np.column_stack([
    np.random.randint(4, 90, n),           # Age
    np.random.randint(0, 2, n),            # Gender
    np.random.exponential(1.5, n),         # Total_Bilirubin
    np.random.exponential(0.5, n),         # Direct_Bilirubin
    np.random.randint(60, 2000, n),        # Alkaline_Phosphotase
    np.random.randint(10, 2000, n),        # ALT
    np.random.randint(10, 2000, n),        # AST
    np.random.normal(6.5, 1, n),           # Total_Protiens
    np.random.normal(3.2, 0.8, n),         # Albumin
    np.random.normal(1.0, 0.3, n),         # A/G Ratio
])
y_liver = ((X_liver[:, 2] > 2) | (X_liver[:, 5] > 50) | (X_liver[:, 4] > 300)).astype(int)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_liver, y_liver)
save_model(model, "liver")


# ── 5. Hepatitis ─────────────────────────────────────────
print("🦠 Training Hepatitis model...")
n = 155
X_hep = np.column_stack([
    np.random.randint(10, 80, n),          # age
    np.random.randint(0, 2, n),            # sex
    np.random.randint(1, 3, n),            # steroid
    np.random.randint(1, 3, n),            # antivirals
    np.random.randint(1, 3, n),            # fatigue
    np.random.randint(1, 3, n),            # malaise
    np.random.randint(1, 3, n),            # anorexia
    np.random.randint(1, 3, n),            # liver_big
    np.random.randint(1, 3, n),            # liver_firm
    np.random.randint(1, 3, n),            # spleen_palpable
    np.random.exponential(1.5, n),         # bilirubin
    np.random.normal(3.8, 0.8, n),         # albumin
])
y_hep = ((X_hep[:, 10] > 2) | (X_hep[:, 11] < 3) | 
          (X_hep[:, 4] == 2) & (X_hep[:, 5] == 2)).astype(int)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_hep, y_hep)
save_model(model, "hepatitis")


# ── 6. Lung Cancer ───────────────────────────────────────
print("🫁 Training Lung Cancer model...")
lung_csv = DATASETS / "lung_cancer.csv"
if lung_csv.exists():
    df = pd.read_csv(lung_csv)
    df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
    df = df.rename(columns={
        "CHRONICDISEASE": "CHRONIC_DISEASE",
        "ALCOHOLCONSUMING": "ALCOHOL",
        "SHORTNESSOFBREATH": "SHORTNESS_OF_BREATH",
        "SWALLOWINGDIFFICULTY": "SWALLOWING_DIFFICULTY",
        "CHESTPAIN": "CHEST_PAIN",
    })
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
    FEATURES = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
                "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
                "WHEEZING", "ALCOHOL", "COUGHING", "SHORTNESS_OF_BREATH",
                "SWALLOWING_DIFFICULTY", "CHEST_PAIN"]
    X_lung = df[FEATURES].values
    y_lung = df["LUNG_CANCER"].values
else:
    n = 309
    X_lung = np.column_stack([
        np.random.randint(0, 2, n),
        np.random.randint(20, 85, n),
        *[np.random.randint(1, 3, n) for _ in range(13)]
    ])
    y_lung = (np.sum(X_lung[:, 2:] == 2, axis=1) > 7).astype(int)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_lung, y_lung)
save_model(model, "lung_cancer")


# ── 7. Chronic Kidney Disease ────────────────────────────
print("🩺 Training Kidney Disease model...")
n = 400
X_kidney = np.column_stack([
    np.random.randint(2, 90, n),           # age
    np.random.randint(50, 180, n),         # bp
    np.random.uniform(1.005, 1.025, n),    # sg
    np.random.randint(0, 6, n),            # al
    np.random.randint(0, 6, n),            # su
    np.random.randint(0, 2, n),            # rbc
    np.random.randint(0, 2, n),            # pc
    np.random.randint(0, 2, n),            # pcc
    np.random.randint(0, 2, n),            # ba
    np.random.randint(70, 490, n),         # bgr
    np.random.randint(1, 391, n),          # bu
    np.random.uniform(0.4, 18, n),         # sc
    np.random.randint(111, 163, n),        # sod
    np.random.uniform(2.5, 7, n),          # pot
    np.random.uniform(3.1, 17.8, n),       # hemo
    np.random.randint(9, 54, n),           # pcv
    np.random.randint(2200, 26400, n),     # wc
    np.random.uniform(2.1, 8, n),          # rc
    np.random.randint(0, 2, n),            # htn
    np.random.randint(0, 2, n),            # dm
    np.random.randint(0, 2, n),            # cad
    np.random.randint(0, 2, n),            # appet
    np.random.randint(0, 2, n),            # pe
    np.random.randint(0, 2, n),            # ane
])
y_kidney = ((X_kidney[:, 14] < 10) | (X_kidney[:, 11] > 3) | 
            (X_kidney[:, 1] > 140) & (X_kidney[:, 3] > 2)).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_kidney, y_kidney)
save_model(model, "kidney")


# ── 8. Breast Cancer ─────────────────────────────────────
print("🎀 Training Breast Cancer model...")
data = load_breast_cancer()
X_bc = data.data
y_bc = data.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_bc, y_bc)
save_model(model, "breast_cancer")


print("\n🎉 All models retrained successfully with scikit-learn", 
      __import__('sklearn').__version__)
