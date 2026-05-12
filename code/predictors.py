"""Numerical disease predictor wrappers around .sav / .joblib model files.

Each predictor describes its input fields so the UI can render forms
generically. If a model file is missing, the predictor reports unavailable.
"""
from pathlib import Path
import joblib
import pickle
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"


def _load(name: str):
    for ext in (".joblib", ".sav", ".pkl"):
        p = MODELS / f"{name}{ext}"
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                with open(p, "rb") as f:
                    return pickle.load(f)
    return None


# Each entry: model filename stem, ordered list of (field, label, type, default, min, max)
SCHEMAS = {
    "Diabetes": {
        "model": "diabetes",
        "fields": [
            ("Pregnancies", "Pregnancies", "int", 1, 0, 20),
            ("Glucose", "Glucose", "float", 110, 0, 300),
            ("BloodPressure", "Blood Pressure", "float", 70, 0, 200),
            ("SkinThickness", "Skin Thickness", "float", 20, 0, 100),
            ("Insulin", "Insulin", "float", 80, 0, 900),
            ("BMI", "BMI", "float", 25.0, 0.0, 70.0),
            ("DiabetesPedigreeFunction", "Diabetes Pedigree Function", "float", 0.5, 0.0, 3.0),
            ("Age", "Age", "int", 35, 1, 120),
        ],
    },
    "Heart Disease": {
        "model": "heart",
        "fields": [
            ("age", "Age", "int", 45, 1, 120),
            ("sex", "Sex (1=M, 0=F)", "int", 1, 0, 1),
            ("cp", "Chest Pain Type (0-3)", "int", 1, 0, 3),
            ("trestbps", "Resting BP", "int", 130, 50, 250),
            ("chol", "Cholesterol", "int", 220, 50, 600),
            ("fbs", "Fasting Sugar >120 (1/0)", "int", 0, 0, 1),
            ("restecg", "Rest ECG (0-2)", "int", 1, 0, 2),
            ("thalach", "Max Heart Rate", "int", 150, 50, 220),
            ("exang", "Exercise Angina (1/0)", "int", 0, 0, 1),
            ("oldpeak", "ST Depression", "float", 1.0, 0.0, 10.0),
            ("slope", "Slope (0-2)", "int", 1, 0, 2),
            ("ca", "Major Vessels (0-3)", "int", 0, 0, 3),
            ("thal", "Thal (0-3)", "int", 2, 0, 3),
        ],
    },
    "Parkinson's": {
        "model": "parkinsons",
        "fields": [
            ("MDVP:Fo(Hz)", "MDVP Fo(Hz)", "float", 154.0, 80.0, 300.0),
            ("MDVP:Fhi(Hz)", "MDVP Fhi(Hz)", "float", 197.0, 100.0, 600.0),
            ("MDVP:Flo(Hz)", "MDVP Flo(Hz)", "float", 116.0, 60.0, 240.0),
            ("MDVP:Jitter(%)", "Jitter %", "float", 0.006, 0.0, 0.05),
            ("MDVP:Shimmer", "Shimmer", "float", 0.03, 0.0, 0.2),
            ("NHR", "NHR", "float", 0.02, 0.0, 1.0),
            ("HNR", "HNR", "float", 21.0, 5.0, 35.0),
            ("RPDE", "RPDE", "float", 0.5, 0.0, 1.0),
            ("DFA", "DFA", "float", 0.7, 0.0, 1.0),
            ("spread1", "spread1", "float", -5.0, -10.0, 0.0),
            ("spread2", "spread2", "float", 0.2, 0.0, 1.0),
            ("D2", "D2", "float", 2.3, 0.0, 5.0),
            ("PPE", "PPE", "float", 0.2, 0.0, 1.0),
        ],
    },"Parkinson's": {
    "model": "parkinsons",
    "fields": [
        ("MDVP:Fo(Hz)", "MDVP Fo(Hz)", "float", 154.0, 80.0, 300.0),
        ("MDVP:Fhi(Hz)", "MDVP Fhi(Hz)", "float", 197.0, 100.0, 600.0),
        ("MDVP:Flo(Hz)", "MDVP Flo(Hz)", "float", 116.0, 60.0, 240.0),
        ("MDVP:Jitter(%)", "Jitter %", "float", 0.006, 0.0, 0.05),
        ("MDVP:Jitter(Abs)", "Jitter Abs", "float", 0.00004, 0.0, 0.001),
        ("MDVP:RAP", "RAP", "float", 0.003, 0.0, 0.03),
        ("MDVP:PPQ", "PPQ", "float", 0.003, 0.0, 0.03),
        ("Jitter:DDP", "Jitter DDP", "float", 0.009, 0.0, 0.09),
        ("MDVP:Shimmer", "Shimmer", "float", 0.03, 0.0, 0.2),
        ("MDVP:Shimmer(dB)", "Shimmer dB", "float", 0.28, 0.0, 2.0),
        ("Shimmer:APQ3", "Shimmer APQ3", "float", 0.015, 0.0, 0.1),
        ("Shimmer:APQ5", "Shimmer APQ5", "float", 0.018, 0.0, 0.1),
        ("MDVP:APQ", "MDVP APQ", "float", 0.024, 0.0, 0.15),
        ("Shimmer:DDA", "Shimmer DDA", "float", 0.045, 0.0, 0.3),
        ("NHR", "NHR", "float", 0.02, 0.0, 1.0),
        ("HNR", "HNR", "float", 21.0, 5.0, 35.0),
        ("RPDE", "RPDE", "float", 0.5, 0.0, 1.0),
        ("DFA", "DFA", "float", 0.7, 0.0, 1.0),
        ("spread1", "spread1", "float", -5.0, -10.0, 0.0),
        ("spread2", "spread2", "float", 0.2, 0.0, 1.0),
        ("D2", "D2", "float", 2.3, 0.0, 5.0),
        ("PPE", "PPE", "float", 0.2, 0.0, 1.0),
    ],
},
    "Liver Disease": {
        "model": "liver",
        "fields": [
            ("Age", "Age", "int", 45, 1, 100),
            ("Gender", "Gender (1=M, 0=F)", "int", 1, 0, 1),
            ("Total_Bilirubin", "Total Bilirubin", "float", 1.0, 0.0, 80.0),
            ("Direct_Bilirubin", "Direct Bilirubin", "float", 0.3, 0.0, 30.0),
            ("Alkaline_Phosphotase", "Alk Phosphatase", "int", 200, 50, 2000),
            ("Alamine_Aminotransferase", "ALT", "int", 35, 1, 2000),
            ("Aspartate_Aminotransferase", "AST", "int", 40, 1, 2000),
            ("Total_Protiens", "Total Proteins", "float", 6.5, 0.0, 10.0),
            ("Albumin", "Albumin", "float", 3.2, 0.0, 6.0),
            ("Albumin_and_Globulin_Ratio", "A/G Ratio", "float", 1.0, 0.0, 3.0),
        ],
    },
    "Hepatitis": {
    "model": "hepatitis",
    "fields": [
        ("age", "Age", "int", 35, 1, 100),
        ("sex", "Sex (1=M, 0=F)", "int", 1, 0, 1),
        ("steroid", "Steroid (1=No, 2=Yes)", "int", 1, 1, 2),
        ("antivirals", "Antivirals (1=No, 2=Yes)", "int", 1, 1, 2),
        ("fatigue", "Fatigue (1=No, 2=Yes)", "int", 1, 1, 2),
        ("malaise", "Malaise (1=No, 2=Yes)", "int", 1, 1, 2),
        ("anorexia", "Anorexia (1=No, 2=Yes)", "int", 1, 1, 2),
        ("liver_big", "Liver Big (1=No, 2=Yes)", "int", 1, 1, 2),
        ("liver_firm", "Liver Firm (1=No, 2=Yes)", "int", 1, 1, 2),
        ("spleen_palpable", "Spleen Palpable (1=No, 2=Yes)", "int", 1, 1, 2),
        ("bilirubin", "Bilirubin", "float", 1.0, 0.0, 8.0),
        ("albumin", "Albumin", "float", 3.8, 0.0, 6.0),
    ],
},
    "Lung Cancer": {
        "model": "lung_cancer",
        "fields": [
            ("GENDER", "Gender (1=M, 0=F)", "int", 1, 0, 1),
            ("AGE", "Age", "int", 55, 1, 100),
            ("SMOKING", "Smoking (1=No, 2=Yes)", "int", 2, 1, 2),
            ("YELLOW_FINGERS", "Yellow Fingers", "int", 1, 1, 2),
            ("ANXIETY", "Anxiety", "int", 1, 1, 2),
            ("PEER_PRESSURE", "Peer Pressure", "int", 1, 1, 2),
            ("CHRONIC_DISEASE", "Chronic Disease", "int", 1, 1, 2),
            ("FATIGUE", "Fatigue", "int", 2, 1, 2),
            ("ALLERGY", "Allergy", "int", 1, 1, 2),
            ("WHEEZING", "Wheezing", "int", 2, 1, 2),
            ("ALCOHOL", "Alcohol", "int", 1, 1, 2),
            ("COUGHING", "Coughing", "int", 2, 1, 2),
            ("SHORTNESS_OF_BREATH", "Shortness of Breath", "int", 2, 1, 2),
            ("SWALLOWING_DIFFICULTY", "Swallowing Difficulty", "int", 1, 1, 2),
            ("CHEST_PAIN", "Chest Pain", "int", 2, 1, 2),
        ],
    },
    "Chronic Kidney Disease": {
    "model": "kidney",
    "fields": [
        ("age", "Age", "int", 50, 1, 100),
        ("bp", "Blood Pressure", "int", 80, 40, 200),
        ("sg", "Specific Gravity", "float", 1.02, 1.0, 1.05),
        ("al", "Albumin (0-5)", "int", 1, 0, 5),
        ("su", "Sugar (0-5)", "int", 0, 0, 5),
        ("rbc", "Red Blood Cells (0=normal,1=abnormal)", "int", 0, 0, 1),
        ("pc", "Pus Cell (0=normal,1=abnormal)", "int", 0, 0, 1),
        ("pcc", "Pus Cell Clumps (0=notpresent,1=present)", "int", 0, 0, 1),
        ("ba", "Bacteria (0=notpresent,1=present)", "int", 0, 0, 1),
        ("bgr", "Blood Glucose Random", "int", 120, 40, 500),
        ("bu", "Blood Urea", "int", 40, 1, 400),
        ("sc", "Serum Creatinine", "float", 1.2, 0.0, 20.0),
        ("sod", "Sodium", "int", 138, 100, 200),
        ("pot", "Potassium", "float", 4.5, 1.0, 10.0),
        ("hemo", "Hemoglobin", "float", 13.0, 3.0, 20.0),
        ("pcv", "Packed Cell Volume", "int", 44, 10, 60),
        ("wc", "White Blood Cell Count", "int", 7800, 2200, 26400),
        ("rc", "Red Blood Cell Count", "float", 5.2, 2.0, 8.0),
        ("htn", "Hypertension (0=no,1=yes)", "int", 0, 0, 1),
        ("dm", "Diabetes Mellitus (0=no,1=yes)", "int", 0, 0, 1),
        ("cad", "Coronary Artery Disease (0=no,1=yes)", "int", 0, 0, 1),
        ("appet", "Appetite (0=good,1=poor)", "int", 0, 0, 1),
        ("pe", "Pedal Edema (0=no,1=yes)", "int", 0, 0, 1),
        ("ane", "Anemia (0=no,1=yes)", "int", 0, 0, 1),
    ],
},
    "Breast Cancer": {
    "model": "breast_cancer",
    "fields": [
        ("mean_radius", "Mean Radius", "float", 14.0, 5.0, 30.0),
        ("mean_texture", "Mean Texture", "float", 19.0, 5.0, 40.0),
        ("mean_perimeter", "Mean Perimeter", "float", 92.0, 40.0, 200.0),
        ("mean_area", "Mean Area", "float", 655.0, 100.0, 2500.0),
        ("mean_smoothness", "Mean Smoothness", "float", 0.1, 0.05, 0.2),
        ("mean_compactness", "Mean Compactness", "float", 0.1, 0.0, 0.35),
        ("mean_concavity", "Mean Concavity", "float", 0.09, 0.0, 0.5),
        ("mean_concave_points", "Mean Concave Points", "float", 0.05, 0.0, 0.2),
        ("mean_symmetry", "Mean Symmetry", "float", 0.18, 0.1, 0.35),
        ("mean_fractal_dimension", "Mean Fractal Dimension", "float", 0.06, 0.04, 0.1),
        ("radius_error", "Radius Error", "float", 0.4, 0.1, 2.5),
        ("texture_error", "Texture Error", "float", 1.2, 0.3, 4.0),
        ("perimeter_error", "Perimeter Error", "float", 2.9, 0.5, 20.0),
        ("area_error", "Area Error", "float", 40.0, 5.0, 250.0),
        ("smoothness_error", "Smoothness Error", "float", 0.007, 0.001, 0.03),
        ("compactness_error", "Compactness Error", "float", 0.025, 0.002, 0.14),
        ("concavity_error", "Concavity Error", "float", 0.03, 0.0, 0.4),
        ("concave_points_error", "Concave Points Error", "float", 0.012, 0.0, 0.05),
        ("symmetry_error", "Symmetry Error", "float", 0.02, 0.007, 0.08),
        ("fractal_dimension_error", "Fractal Dimension Error", "float", 0.004, 0.0008, 0.03),
        ("worst_radius", "Worst Radius", "float", 16.0, 7.0, 40.0),
        ("worst_texture", "Worst Texture", "float", 25.0, 10.0, 50.0),
        ("worst_perimeter", "Worst Perimeter", "float", 107.0, 50.0, 260.0),
        ("worst_area", "Worst Area", "float", 880.0, 150.0, 4300.0),
        ("worst_smoothness", "Worst Smoothness", "float", 0.13, 0.07, 0.25),
        ("worst_compactness", "Worst Compactness", "float", 0.25, 0.02, 1.1),
        ("worst_concavity", "Worst Concavity", "float", 0.27, 0.0, 1.3),
        ("worst_concave_points", "Worst Concave Points", "float", 0.11, 0.0, 0.3),
        ("worst_symmetry", "Worst Symmetry", "float", 0.29, 0.15, 0.7),
        ("worst_fractal_dimension", "Worst Fractal Dimension", "float", 0.08, 0.05, 0.25),
    ],
},
}


def get_predictor(disease: str):
    schema = SCHEMAS[disease]
    model = _load(schema["model"])
    return model, schema["fields"]


def run_prediction(model, values: list[float]):
    arr = np.array(values, dtype=float).reshape(1, -1)
    pred = model.predict(arr)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(arr)[0]
            conf = float(np.max(proba))
        except Exception:
            conf = None
    return int(pred) if not isinstance(pred, str) else pred, conf
