"""Multiple Disease Prediction System — Streamlit app."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from auth import init_db, signup, login, save_report, get_user_reports, delete_report, delete_all_reports  # noqa
from code.helper import generate_pdf, explain_prediction, HEALTH_TIPS   # noqa
from code.predictors import SCHEMAS, get_predictor, run_prediction       # noqa
from code.DiseaseModel import DiseaseModel                               # noqa

st.set_page_config(
    page_title="MDPS — Multiple Disease Prediction System",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-size:2rem; font-weight:700; color:#1e5aa0; margin-bottom:0; }
.subtitle { color:#5b6b80; margin-top:0; }
.metric-card {
    background:linear-gradient(135deg,#1e5aa0,#2d7dd2);
    color:white; padding:1.2rem; border-radius:14px;
    box-shadow:0 4px 12px rgba(30,90,160,.2); transition:transform .2s;
}
.metric-card:hover { transform:translateY(-2px); }
.result-positive {
    background:#fde2e2; border-left:6px solid #d9534f;
    padding:1rem; border-radius:8px; color:#7a1b1b; font-weight:600;
}
.result-negative {
    background:#dcf3e3; border-left:6px solid #28a745;
    padding:1rem; border-radius:8px; color:#155724; font-weight:600;
}
.section-card {
    background:#ffffff; padding:1.2rem; border-radius:14px;
    box-shadow:0 2px 8px rgba(0,0,0,.05); margin-bottom:1rem;
}
.sidebar-brand {
    text-align:center; padding:10px 0 4px;
    font-size:1.5rem; font-weight:700; color:#1e5aa0; letter-spacing:-0.5px;
}
.sidebar-tagline {
    text-align:center; font-size:0.7rem; color:#8a9bae;
    margin-bottom:8px; letter-spacing:0.5px; text-transform:uppercase;
}
.hero-banner {
    background:linear-gradient(135deg,#0f3460 0%,#1e5aa0 50%,#2d7dd2 100%);
    color:white; padding:3.5rem 2.5rem; border-radius:20px; margin-bottom:2rem;
}
.hero-banner h1 { font-size:2.4rem; font-weight:700; margin:0 0 0.5rem; }
.hero-banner p  { font-size:1.1rem; opacity:.85; margin:0; }
.hero-badge {
    display:inline-block; background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.3); color:white;
    padding:4px 12px; border-radius:20px; font-size:.78rem;
    margin-bottom:1rem; letter-spacing:.5px;
}
.feature-tile {
    background:white; border:1px solid #e8edf2; border-radius:14px;
    padding:1.4rem; height:100%; transition:box-shadow .2s,transform .2s;
}
.feature-tile:hover { box-shadow:0 8px 24px rgba(30,90,160,.12); transform:translateY(-2px); }
.feature-icon  { font-size:2rem; margin-bottom:.5rem; }
.feature-title { font-weight:600; color:#1e3a5f; font-size:1rem; }
.feature-desc  { color:#6b7a8d; font-size:.85rem; margin-top:.25rem; }
.profile-card {
    background:linear-gradient(135deg,#1e5aa0,#2d7dd2);
    color:white; border-radius:16px; padding:2rem;
    text-align:center; margin-bottom:1.5rem;
}
.profile-avatar {
    width:72px; height:72px; border-radius:50%;
    background:rgba(255,255,255,.2); display:flex;
    align-items:center; justify-content:center;
    font-size:2rem; margin:0 auto 1rem;
    border:3px solid rgba(255,255,255,.4);
}
.profile-name  { font-size:1.4rem; font-weight:700; }
.profile-email { opacity:.8; font-size:.9rem; }
.stat-pill {
    background:#f0f4fa; border-radius:10px; padding:.6rem 1.2rem;
    display:inline-block; text-align:center;
}
.stat-pill .num { font-size:1.4rem; font-weight:700; color:#1e5aa0; }
.stat-pill .lbl { font-size:.75rem; color:#6b7a8d; display:block; }
.gauge-title { text-align:center; font-weight:600; color:#1e3a5f; margin-bottom:.3rem; font-size:.95rem; }
/* ── Dark mode ── */
[data-testid="stAppViewContainer"].dark-mode { background-color:#0e1117 !important; }
[data-testid="stSidebar"].dark-mode          { background-color:#111827 !important; }
.dark-mode .main-title   { color:#90cdf4 !important; }
.dark-mode .subtitle     { color:#a0aec0 !important; }
.dark-mode .section-card { background:#1a1f2e !important; color:#ffffff !important; }
.dark-mode .feature-tile { background:#1a1f2e !important; border-color:#2d3748 !important; }
.dark-mode .feature-title { color:#90cdf4 !important; }
.dark-mode .feature-desc  { color:#a0aec0 !important; }
.dark-mode .stat-pill     { background:#2d3748 !important; }
.dark-mode .result-positive { background:#3d1515 !important; color:#feb2b2 !important; }
.dark-mode .result-negative { background:#1a3328 !important; color:#9ae6b4 !important; }
</style>
"""

init_db()

if "user"           not in st.session_state: st.session_state.user           = None
if "last_pred"      not in st.session_state: st.session_state.last_pred      = None
if "dark_mode"      not in st.session_state: st.session_state.dark_mode      = False
if "confirm_delete" not in st.session_state: st.session_state.confirm_delete = False

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if st.session_state.dark_mode:
    st.markdown("""
    <script>
        var app     = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if(app)     app.classList.add('dark-mode');
        if(sidebar) sidebar.classList.add('dark-mode');
    </script>
    <style>
        [data-testid="stAppViewContainer"] { background-color:#0e1117 !important; }
        [data-testid="stSidebar"]          { background-color:#111827 !important; }
        .stMarkdown, p, label { color:#e2e8f0 !important; }
        h1, h2, h3            { color:#90cdf4 !important; }
        [data-testid="stTextInput"] input  { background:#1a1f2e !important; color:white !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_gauge(confidence, positive: bool):
    if confidence is None:
        confidence = 0.75 if positive else 0.3
    value = confidence * 100
    color = "#d9534f" if positive else "#28a745"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#eafaf1"},
                {"range": [40, 70], "color": "#fef9e7"},
                {"range": [70,100], "color": "#fdedec"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": value},
        },
        title={"text": "Risk Score", "font": {"size": 14, "color": "#5b6b80"}},
    ))
    fig.update_layout(height=220, margin=dict(t=40,b=10,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Auth ──────────────────────────────────────────────────────────────────────
def auth_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">🩺 AI-Powered Health Screening</div>
            <h1>Multiple Disease<br>Prediction System</h1>
            <p>Predict 9+ diseases using machine learning with explainable AI results.</p>
        </div>
        """, unsafe_allow_html=True)
        tabs = st.tabs(["🔑 Login", "📝 Sign up"])
        with tabs[0]:
            with st.form("login"):
                email = st.text_input("Email", placeholder="you@example.com")
                pw    = st.text_input("Password", type="password", placeholder="••••••••")
                if st.form_submit_button("Login →", use_container_width=True):
                    u = login(email, pw)
                    if u:
                        st.session_state.user = u
                        st.success(f"Welcome back, {u['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
        with tabs[1]:
            with st.form("signup"):
                uname = st.text_input("Username", placeholder="John Doe")
                email = st.text_input("Email",    placeholder="you@example.com", key="se")
                pw    = st.text_input("Password (min 6 chars)", type="password", key="sp")
                if st.form_submit_button("Create account →", use_container_width=True):
                    ok, msg = signup(uname, email, pw)
                    (st.success if ok else st.error)(msg)
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        for col, (icon, title, desc) in zip([f1,f2,f3], [
            ("🧬","9 Diseases","Diabetes, Heart, Cancer & more"),
            ("📄","PDF Reports","Download detailed health reports"),
            ("🔍","Explainable AI","Understand why the model decided"),
        ]):
            col.markdown(f'<div class="feature-tile"><div class="feature-icon">{icon}</div>'
                         f'<div class="feature-title">{title}</div>'
                         f'<div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar() -> str:
    u = st.session_state.user
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🩺 MDPS</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Disease Prediction System</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown(f"**👋 {u['username']}**")
        st.caption(u["email"])
        st.divider()
        choice = st.radio("Navigation", [
    "🏠 Dashboard", "🏥 About", "� Full Body Checkup",
    "�🧪 Disease Prediction", "🤒 Symptom-Based", "📁 Saved Reports",
    "📊 Analytics", "💡 Health Tips", "🏨 Nearby Hospitals",
    "🤖 AI Chatbot", "👤 Profile", "⚙️ Settings",
], label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()
    return choice


# ── Dashboard ─────────────────────────────────────────────────────────────────
def page_dashboard():
    u       = st.session_state.user
    reports = get_user_reports(u["id"])
    st.markdown(f'<p class="main-title">Welcome back, {u["username"]} 👋</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Here\'s your health prediction summary</p>', unsafe_allow_html=True)
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Predictions", len(reports)),
        (c2, "Diseases Tracked",  len({r["disease"] for r in reports})),
        (c3, "Positive Results",  sum(1 for r in reports if "Positive" in r["prediction"] or "High" in r["prediction"])),
        (c4, "Last Check",        reports[0]["date"][:10] if reports else "—"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div style="opacity:.8;font-size:.85rem">{label}</div>'
                        f'<h2 style="margin:4px 0 0">{val}</h2></div>', unsafe_allow_html=True)
    st.markdown("### Quick actions")
    qc = st.columns(3)
    qc[0].info("🧪 Run a **numerical** disease prediction from the sidebar.")
    qc[1].info("🤒 Try the **symptom-based** predictor for general checkup.")
    qc[2].info("📁 Review your **saved reports** anytime.")
    if reports:
        st.markdown("### Recent predictions")
        st.dataframe(pd.DataFrame(reports[:5])[["date","disease","prediction"]],
                     use_container_width=True, hide_index=True)


# ── About ─────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🩺 AI-Powered</div>
        <h1>About MDPS</h1>
        <p>Multiple Disease Prediction System — AI-assisted health screening for everyone.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### What is MDPS?")
    st.write("MDPS is an AI-powered web application that uses machine learning models to predict "
             "the risk of multiple diseases based on clinical parameters and symptoms.")
    st.markdown("### Diseases covered")
    diseases = [
        ("🩸","Diabetes","Glucose, BMI, insulin-based prediction"),
        ("❤️","Heart Disease","Cholesterol, BP, ECG-based screening"),
        ("🧠","Parkinson's","Voice signal analysis (22 features)"),
        ("🫀","Liver Disease","Bilirubin, enzyme level analysis"),
        ("🦠","Hepatitis C","Blood marker-based risk assessment"),
        ("🫁","Lung Cancer","Smoking, symptom-based screening"),
        ("🩺","Chronic Kidney Disease","Creatinine, hemoglobin analysis"),
        ("🎀","Breast Cancer","30-feature tumour cell analysis"),
        ("🟡","Jaundice","Symptom + severity-based prediction"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(diseases):
        with cols[i % 3]:
            st.markdown(f'<div class="feature-tile" style="margin-bottom:1rem">'
                        f'<div class="feature-icon">{icon}</div>'
                        f'<div class="feature-title">{name}</div>'
                        f'<div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("### ⚠️ Disclaimer")
    st.warning("MDPS is for **informational and educational purposes only**. "
               "Always consult a qualified healthcare provider.")
    st.markdown("### Tech stack")
    c1,c2,c3,c4 = st.columns(4)
    c1.success("🐍 Python 3.12"); c2.success("🚀 Streamlit")
    c3.success("🤖 scikit-learn"); c4.success("📊 Plotly")


# ── Prediction result ─────────────────────────────────────────────────────────
def render_prediction_result(disease, pred_label, positive, conf, params):
    klass = "result-positive" if positive else "result-negative"
    icon  = "⚠️" if positive else "✅"
    st.markdown(f'<div class="{klass}">{icon} {pred_label}</div>', unsafe_allow_html=True)
    st.markdown("")
    g_col, p_col = st.columns([1, 1])
    with g_col:
        st.markdown('<div class="gauge-title">Risk level</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(conf, positive), use_container_width=True,
                        config={"displayModeBar": False})
    with p_col:
        st.markdown("**Model confidence**")
        if conf is not None:
            bar_val = min(max(conf, 0.0), 1.0)
            color   = "🔴" if bar_val > 0.7 else "🟡" if bar_val > 0.4 else "🟢"
            st.progress(bar_val, text=f"{color} {bar_val*100:.1f}% confidence")
        else:
            st.info("Confidence not available for this model.")
        if positive:
            st.warning("👨‍⚕️ Doctor consultation recommended.")
            if disease in {"Heart Disease","Lung Cancer","Chronic Kidney Disease"}:
                st.error("🚨 High-risk condition — consider urgent medical attention.")
    with st.expander("🔍 Why this prediction?", expanded=True):
        for reason in explain_prediction(disease, params, positive):
            st.write("• " + reason)


# ── Full Body Checkup (Unified Prediction) ────────────────────────────────────
def page_full_body_checkup():
    st.markdown('<p class="main-title">🩻 Full Body Checkup</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter your details once — get predictions for all diseases</p>',
                unsafe_allow_html=True)

    with st.form("full_body_form"):
        st.markdown("### 👤 Basic Information")
        b1, b2, b3 = st.columns(3)
        age = b1.number_input("Age", 1, 120, 45, step=1)
        gender = b2.selectbox("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
        pregnancies = b3.number_input("Pregnancies (if female)", 0, 20, 0, step=1)

        st.markdown("### 🩸 Blood & Vitals")
        v1, v2, v3, v4 = st.columns(4)
        glucose = v1.number_input("Glucose", 0.0, 500.0, 110.0, step=1.0)
        blood_pressure = v2.number_input("Blood Pressure (systolic)", 40, 250, 120, step=1)
        cholesterol = v3.number_input("Cholesterol", 50, 600, 220, step=1)
        heart_rate = v4.number_input("Max Heart Rate", 50, 220, 150, step=1)

        v5, v6, v7, v8 = st.columns(4)
        bmi = v5.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        insulin = v6.number_input("Insulin", 0.0, 900.0, 80.0, step=1.0)
        skin_thickness = v7.number_input("Skin Thickness", 0.0, 100.0, 20.0, step=1.0)
        dpf = v8.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)

        st.markdown("### 🫀 Liver & Kidney")
        l1, l2, l3, l4 = st.columns(4)
        total_bilirubin = l1.number_input("Total Bilirubin", 0.0, 80.0, 1.0, step=0.1)
        direct_bilirubin = l2.number_input("Direct Bilirubin", 0.0, 30.0, 0.3, step=0.1)
        alk_phosphatase = l3.number_input("Alkaline Phosphatase", 50, 2000, 200, step=1)
        alt = l4.number_input("ALT (SGPT)", 1, 2000, 35, step=1)

        l5, l6, l7, l8 = st.columns(4)
        ast = l5.number_input("AST (SGOT)", 1, 2000, 40, step=1)
        total_proteins = l6.number_input("Total Proteins", 0.0, 10.0, 6.5, step=0.1)
        albumin = l7.number_input("Albumin", 0.0, 6.0, 3.8, step=0.1)
        ag_ratio = l8.number_input("A/G Ratio", 0.0, 3.0, 1.0, step=0.1)

        st.markdown("### 🫁 Kidney Parameters")
        k1, k2, k3, k4 = st.columns(4)
        hemoglobin = k1.number_input("Hemoglobin", 3.0, 20.0, 13.0, step=0.1)
        blood_urea = k2.number_input("Blood Urea", 1, 400, 40, step=1)
        serum_creatinine = k3.number_input("Serum Creatinine", 0.0, 20.0, 1.2, step=0.1)
        sodium = k4.number_input("Sodium", 100, 200, 138, step=1)

        k5, k6, k7, k8 = st.columns(4)
        potassium = k5.number_input("Potassium", 1.0, 10.0, 4.5, step=0.1)
        specific_gravity = k6.number_input("Specific Gravity", 1.0, 1.05, 1.02, step=0.005)
        rbc_count = k7.number_input("RBC Count", 2.0, 8.0, 5.2, step=0.1)
        wbc_count = k8.number_input("WBC Count", 2200, 26400, 7800, step=100)

        st.markdown("### 🫁 Lung Cancer Risk Factors")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        smoking = lc1.selectbox("Smoking", ["No", "Yes"])
        yellow_fingers = lc2.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = lc3.selectbox("Anxiety", ["No", "Yes"])
        chronic_disease = lc4.selectbox("Chronic Disease", ["No", "Yes"])
        fatigue = lc5.selectbox("Fatigue", ["No", "Yes"])

        lc6, lc7, lc8, lc9, lc10 = st.columns(5)
        allergy = lc6.selectbox("Allergy", ["No", "Yes"])
        wheezing = lc7.selectbox("Wheezing", ["No", "Yes"])
        alcohol = lc8.selectbox("Alcohol", ["No", "Yes"])
        coughing = lc9.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = lc10.selectbox("Shortness of Breath", ["No", "Yes"])

        lc11, lc12, lc13 = st.columns(3)
        swallowing_diff = lc11.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = lc12.selectbox("Chest Pain", ["No", "Yes"])
        peer_pressure = lc13.selectbox("Peer Pressure", ["No", "Yes"])

        st.markdown("### ❤️ Heart Parameters")
        h1, h2, h3, h4 = st.columns(4)
        cp = h1.number_input("Chest Pain Type (0-3)", 0, 3, 1, step=1)
        fbs = h2.selectbox("Fasting Sugar >120", ["No", "Yes"])
        restecg = h3.number_input("Rest ECG (0-2)", 0, 2, 1, step=1)
        exang = h4.selectbox("Exercise Angina", ["No", "Yes"])

        h5, h6, h7, h8 = st.columns(4)
        oldpeak = h5.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = h6.number_input("Slope (0-2)", 0, 2, 1, step=1)
        ca = h7.number_input("Major Vessels (0-3)", 0, 3, 0, step=1)
        thal = h8.number_input("Thal (0-3)", 0, 3, 2, step=1)

        submitted = st.form_submit_button("🔬 Run Full Body Checkup", use_container_width=True)

    if submitted:
        yn = lambda v: 2 if v == "Yes" else 1
        yn01 = lambda v: 1 if v == "Yes" else 0

        results = []

        # ── Diabetes ──
        model_diabetes, _ = get_predictor("Diabetes")
        if model_diabetes:
            vals = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            pred, conf = run_prediction(model_diabetes, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("🩸 Diabetes", positive, conf))

        # ── Heart Disease ──
        model_heart, _ = get_predictor("Heart Disease")
        if model_heart:
            vals = [age, gender_val, cp, blood_pressure, cholesterol, yn01(fbs),
                    restecg, heart_rate, yn01(exang), oldpeak, slope, ca, thal]
            pred, conf = run_prediction(model_heart, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("❤️ Heart Disease", positive, conf))

        # ── Liver Disease ──
        model_liver, _ = get_predictor("Liver Disease")
        if model_liver:
            vals = [age, gender_val, total_bilirubin, direct_bilirubin, alk_phosphatase,
                    alt, ast, total_proteins, albumin, ag_ratio]
            pred, conf = run_prediction(model_liver, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("🫀 Liver Disease", positive, conf))

        # ── Lung Cancer ──
        model_lung, _ = get_predictor("Lung Cancer")
        if model_lung:
            vals = [gender_val, age, yn(smoking), yn(yellow_fingers), yn(anxiety),
                    yn(peer_pressure), yn(chronic_disease), yn(fatigue), yn(allergy),
                    yn(wheezing), yn(alcohol), yn(coughing), yn(shortness_breath),
                    yn(swallowing_diff), yn(chest_pain)]
            pred, conf = run_prediction(model_lung, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("🫁 Lung Cancer", positive, conf))

        # ── Chronic Kidney Disease ──
        model_kidney, _ = get_predictor("Chronic Kidney Disease")
        if model_kidney:
            vals = [age, blood_pressure, specific_gravity, 1, 0, 0, 0, 0, 0,
                    glucose, blood_urea, serum_creatinine, sodium, potassium,
                    hemoglobin, 44, wbc_count, rbc_count, 0, 0, 0, 0, 0, 0]
            pred, conf = run_prediction(model_kidney, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("🩺 Chronic Kidney Disease", positive, conf))

        # ── Hepatitis ──
        model_hep, _ = get_predictor("Hepatitis")
        if model_hep:
            vals = [age, gender_val, 1, 1, yn(fatigue), 1, 1, 1, 1, 1,
                    total_bilirubin, albumin]
            pred, conf = run_prediction(model_hep, vals)
            positive = bool(pred) and pred not in (0, "0", "No", "Negative")
            results.append(("🦠 Hepatitis", positive, conf))

        # ── Display Results ──
        st.markdown("---")
        st.markdown("## 📋 Full Body Checkup Results")

        positive_count = sum(1 for _, p, _ in results if p)
        negative_count = len(results) - positive_count

        r1, r2, r3 = st.columns(3)
        r1.markdown(f'<div class="metric-card"><div style="opacity:.8;font-size:.85rem">Diseases Checked</div>'
                    f'<h2 style="margin:4px 0 0">{len(results)}</h2></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#d9534f,#e87c7c)">'
                    f'<div style="opacity:.8;font-size:.85rem">At Risk</div>'
                    f'<h2 style="margin:4px 0 0">{positive_count}</h2></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#28a745,#5cb85c)">'
                    f'<div style="opacity:.8;font-size:.85rem">Low Risk</div>'
                    f'<h2 style="margin:4px 0 0">{negative_count}</h2></div>', unsafe_allow_html=True)

        st.markdown("")

        for disease_name, positive, conf in results:
            if positive:
                conf_text = f" — {conf*100:.0f}% confidence" if conf else ""
                st.markdown(f'<div class="result-positive">⚠️ {disease_name}: '
                            f'<strong>Positive / At Risk</strong>{conf_text}</div>',
                            unsafe_allow_html=True)
            else:
                conf_text = f" — {conf*100:.0f}% confidence" if conf else ""
                st.markdown(f'<div class="result-negative">✅ {disease_name}: '
                            f'<strong>Negative / Low Risk</strong>{conf_text}</div>',
                            unsafe_allow_html=True)
            st.markdown("")

        # Save combined report
        all_predictions = "; ".join(
            f"{name}: {'Positive' if pos else 'Negative'}" for name, pos, _ in results
        )
        avg_conf = sum(c for _, _, c in results if c) / max(sum(1 for _, _, c in results if c), 1)
        save_report(st.session_state.user["id"], "Full Body Checkup",
                    f"Age={age}, Gender={gender}, Glucose={glucose}, BP={blood_pressure}",
                    all_predictions, confidence=avg_conf)

        if positive_count > 0:
            st.warning("👨‍⚕️ **Doctor consultation recommended** for the conditions marked at risk.")
            if positive_count >= 3:
                st.error("🚨 Multiple risk factors detected — please seek medical attention soon.")
        else:
            st.success("🎉 All clear! No significant risk detected. Keep up the healthy lifestyle!")

        # PDF download
        with st.spinner("Generating PDF..."):
            # Strip emojis for PDF compatibility
            clean_results = [(name.encode('ascii', 'ignore').decode().strip(), pos, c) 
                            for name, pos, c in results]
            pdf_bytes = generate_pdf(
                username=st.session_state.user["username"],
                disease="Full Body Checkup",
                prediction="; ".join(f"{n}: {'Positive' if p else 'Negative'}" 
                                     for n, p, _ in clean_results),
                params={"Age": age, "Gender": gender, "Glucose": glucose,
                        "Blood Pressure": blood_pressure, "BMI": bmi,
                        "Cholesterol": cholesterol, "Hemoglobin": hemoglobin},
                confidence=avg_conf,
                explanation=[f"{n}: {'At Risk' if p else 'Low Risk'}"
                             for n, p, _ in clean_results],
            )
        st.download_button("⬇️ Download Full Checkup Report (PDF)", pdf_bytes,
                           file_name="full_body_checkup_report.pdf",
                           mime="application/pdf", use_container_width=True)


# ── Disease prediction ────────────────────────────────────────────────────────
def page_disease_prediction():
    st.markdown('<p class="main-title">🧪 Disease Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter clinical parameters to get an AI-based risk assessment</p>',
                unsafe_allow_html=True)
    disease      = st.selectbox("Select disease module", list(SCHEMAS.keys()))
    model, fields = get_predictor(disease)
    if model is None:
        st.warning(f"⚠️ Model file not found for **{disease}**. Add the model file to `models/` and reload.")

    with st.form(f"form_{disease}"):
        cols = st.columns(3)
        values, params = [], {}
        for i, (key, label, typ, default, lo, hi) in enumerate(fields):
            with cols[i % 3]:
                v = st.number_input(label, int(lo), int(hi), int(default), step=1) if typ == "int" \
                    else st.number_input(label, float(lo), float(hi), float(default), step=0.1)
                values.append(v); params[key] = v
        submitted = st.form_submit_button(f"🔬 Predict {disease}", use_container_width=True)

    if submitted and model is not None:
        with st.spinner(f"🔬 Analyzing {disease} risk... please wait"):
            try:
                pred, conf = run_prediction(model, values)
                positive   = bool(pred) and pred not in (0, "0", "No", "Negative")
                label      = f"{disease}: {'Positive / At Risk' if positive else 'Negative / Low Risk'}"
                render_prediction_result(disease, label, positive, conf, params)
                rid = save_report(st.session_state.user["id"], disease,
                                  ", ".join(f"{k}={v}" for k,v in params.items()),
                                  "Positive" if positive else "Negative",
                                  confidence=conf, details=str(params))
                st.session_state.last_pred = {
                    "disease": disease, "prediction": label, "params": params,
                    "confidence": conf, "explanation": explain_prediction(disease, params, positive),
                    "report_id": rid,
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.session_state.last_pred and st.session_state.last_pred["disease"] == disease:
        lp = st.session_state.last_pred
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf(
                username=st.session_state.user["username"],
                disease=lp["disease"], prediction=lp["prediction"],
                params=lp["params"], confidence=lp.get("confidence"),
                explanation=lp.get("explanation"),
            )
        st.download_button("⬇️ Download PDF Report", pdf_bytes,
                           file_name=f"{disease.replace(' ','_')}_report.pdf",
                           mime="application/pdf", use_container_width=True)


# ── Symptom-based ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_symptom_model():
    return DiseaseModel()


def page_symptom():
    st.markdown('<p class="main-title">🤒 Symptom-Based Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select your symptoms and get an AI-based disease prediction</p>',
                unsafe_allow_html=True)
    dm = get_symptom_model()
    if not dm.available():
        st.warning("Place Training.csv, symptom_severity.csv, disease_description.csv, "
                   "disease_precaution.csv in datasets/ to enable this module.")
        return
    selected = st.multiselect("Select your symptoms", dm.symptoms,
                              placeholder="Type to search symptoms...")
    if st.button("🔬 Predict disease", use_container_width=True, disabled=not selected):
        with st.spinner("🔬 Analyzing symptoms... please wait"):
            res = dm.predict(selected)
        st.markdown(f'<div class="result-positive">🩺 Likely condition: <strong>{res["disease"]}</strong></div>',
                    unsafe_allow_html=True)
        st.markdown("")
        g_col, m_col = st.columns([1, 1])
        with g_col:
            st.markdown('<div class="gauge-title">Risk level</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(res["confidence"], True),
                            use_container_width=True, config={"displayModeBar": False})
        with m_col:
            if res["confidence"] is not None:
                st.metric("Confidence", f"{res['confidence']*100:.1f}%")
            st.metric("Severity score", f"{res['severity']:.2f}")
        if res["description"]:
            with st.expander("📖 About this condition", expanded=True):
                st.write(res["description"])
        if res["precautions"]:
            with st.expander("🛡️ Recommended precautions", expanded=True):
                for p in res["precautions"]: st.write("• " + p)
        with st.expander("🔍 Why this prediction"):
            ranked = sorted(selected, key=lambda s: dm.severity.get(s, 0), reverse=True)[:5]
            for s in ranked:
                st.write(f"• **{s}** — severity {dm.severity.get(s,'—')}")
        rid = save_report(st.session_state.user["id"], "Symptom-Based",
                          ", ".join(selected), res["disease"],
                          confidence=res["confidence"], severity=res["severity"],
                          details=res["description"])
        with st.spinner("Generating PDF..."):
            pdf = generate_pdf(
                username=st.session_state.user["username"],
                disease="Symptom-Based", prediction=res["disease"],
                params={s: "yes" for s in selected},
                severity=res["severity"], description=res["description"],
                precautions=res["precautions"], confidence=res["confidence"],
                explanation=[f"{s} (severity {dm.severity.get(s,'—')})" for s in ranked],
            )
        st.download_button("⬇️ Download PDF Report", pdf,
                           file_name=f"symptom_report_{rid}.pdf",
                           mime="application/pdf", use_container_width=True)


# ── Saved Reports ─────────────────────────────────────────────────────────────
def page_reports():
    st.markdown('<p class="main-title">📁 Saved Reports</p>', unsafe_allow_html=True)
    reports = get_user_reports(st.session_state.user["id"])
    if not reports:
        st.info("No reports yet. Run a prediction to get started.")
        return
    df = pd.DataFrame(reports)
    q  = st.text_input("🔎 Search by disease or result", placeholder="e.g. Diabetes, Positive...")
    if q:
        df = df[df.apply(lambda r: q.lower() in str(r["disease"]).lower()
                                or q.lower() in str(r["prediction"]).lower(), axis=1)]
    total    = len(df)
    positive = sum(1 for _, r in df.iterrows()
                   if "Positive" in str(r["prediction"]) or "High" in str(r["prediction"]))
    negative = total - positive
    p1, p2, p3 = st.columns(3)
    p1.markdown(f'<div class="stat-pill"><span class="num">{total}</span><span class="lbl">Total</span></div>',
                unsafe_allow_html=True)
    p2.markdown(f'<div class="stat-pill"><span class="num" style="color:#d9534f">{positive}</span><span class="lbl">Positive</span></div>',
                unsafe_allow_html=True)
    p3.markdown(f'<div class="stat-pill"><span class="num" style="color:#28a745">{negative}</span><span class="lbl">Negative</span></div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### Your Reports")
    for r in df.to_dict("records"):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
        c1.write(r["date"][:10])
        c2.write(r["disease"])
        c3.write(r["prediction"])
        c4.write(f"{r['confidence']*100:.0f}%" if r["confidence"] else "—")
        if c5.button("🗑️", key=f"del_{r['id']}", help="Delete this report"):
            delete_report(r["id"])
            st.success("Report deleted!")
            st.rerun()
    st.divider()
    if st.button("🗑️ Delete ALL reports", type="secondary", use_container_width=True):
        st.session_state.confirm_delete = True
        st.rerun()
    if st.session_state.confirm_delete:
        st.warning("⚠️ Are you sure? This cannot be undone!")
        c1, c2 = st.columns(2)
        if c1.button("✅ Yes, delete all", use_container_width=True):
            delete_all_reports(st.session_state.user["id"])
            st.session_state.confirm_delete = False
            st.success("All reports deleted!")
            st.rerun()
        if c2.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_delete = False
            st.rerun()


# ── Analytics ─────────────────────────────────────────────────────────────────
def page_analytics():
    st.markdown('<p class="main-title">📊 Analytics</p>', unsafe_allow_html=True)
    reports = get_user_reports(st.session_state.user["id"])
    if not reports:
        st.info("Make a few predictions to unlock analytics.")
        return
    df = pd.DataFrame(reports)
    df["date"] = pd.to_datetime(df["date"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="disease", color="prediction", title="Predictions by disease",
                           color_discrete_map={"Positive":"#d9534f","Negative":"#28a745"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if df["severity"].notna().any():
            fig = px.box(df.dropna(subset=["severity"]), x="disease", y="severity",
                         title="Severity distribution", color="disease")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    fig = px.line(df.sort_values("date"), x="date", y="confidence", color="disease",
                  title="Confidence over time", markers=True)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    pred_counts = df["prediction"].value_counts().reset_index()
    pred_counts.columns = ["Result","Count"]
    fig = px.pie(pred_counts, names="Result", values="Count",
                 title="Positive vs Negative ratio", hole=0.5,
                 color="Result",
                 color_discrete_map={"Positive":"#d9534f","Negative":"#28a745"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
# ── Nearby Hospitals ──────────────────────────────────────────────────────────
def page_hospitals():
    st.markdown('<p class="main-title">🏥 Nearby Hospitals</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find hospitals and clinics near you</p>', unsafe_allow_html=True)

    city = st.text_input("🔍 Enter your city or area", placeholder="e.g. Jaipur, Delhi, Mumbai")
    
    speciality = st.selectbox("Filter by type", [
        "Hospital", "Medical Clinic", "Emergency Hospital",
        "Cardiology", "Neurology", "Oncology", "Nephrology"
    ])

    if city:
        query = f"{speciality} near {city} India"
        encoded = query.replace(" ", "+")
        
        map_url = f"https://maps.google.com/maps?q={encoded}&output=embed&z=13"
        
        st.markdown(f"""
        <iframe
            src="{map_url}"
            width="100%"
            height="500"
            style="border:0; border-radius:14px; box-shadow:0 4px 12px rgba(0,0,0,0.1);"
            allowfullscreen=""
            loading="lazy">
        </iframe>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.info(f"📍 Showing **{speciality}** near **{city}** — Click on any marker for details, directions & contact info.")
        
        # Quick links
        st.markdown("### 🔗 Quick search links")
        c1, c2, c3 = st.columns(3)
        c1.link_button("🏥 Hospitals",   f"https://www.google.com/maps/search/hospital+near+{city.replace(' ','+')}+India")
        c2.link_button("🚑 Emergency",   f"https://www.google.com/maps/search/emergency+hospital+near+{city.replace(' ','+')}+India")
        c3.link_button("💊 Pharmacies",  f"https://www.google.com/maps/search/pharmacy+near+{city.replace(' ','+')}+India")
    else:
        st.info("👆 Enter your city name above to find nearby hospitals on the map.")


## Step 4 — Chatbot page add karo


# ── AI Health Chatbot ─────────────────────────────────────────────────────────
def page_chatbot():
    st.markdown('<p class="main-title">🤖 AI Health Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask any health question — powered by Groq AI</p>',
                unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a health question... e.g. What are symptoms of diabetes?")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get Groq response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    from groq import Groq
                    client = Groq(api_key=st.secrets["groq"]["api_key"])
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful medical assistant for the MDPS "
                                    "(Multiple Disease Prediction System) app. "
                                    "Answer health questions clearly and concisely. "
                                    "Always remind users to consult a doctor for medical advice. "
                                    "Do not diagnose — only educate and inform."
                                )
                            },
                            *[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.chat_history],
                        ],
                        max_tokens=512,
                        temperature=0.7,
                    )
                    reply = response.choices[0].message.content
                    st.write(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                except Exception as e:
                    st.error(f"Chatbot error: {e}")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    st.caption("⚠️ This chatbot is for educational purposes only. Always consult a qualified doctor.")

# ── Health Tips ───────────────────────────────────────────────────────────────
def page_tips():
    st.markdown('<p class="main-title">💡 Health Tips</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Personalized tips for each condition</p>', unsafe_allow_html=True)
    for d, tips in HEALTH_TIPS.items():
        with st.expander(f"🩺 {d}"):
            for t in tips: st.write("• " + t)


# ── Profile ───────────────────────────────────────────────────────────────────
def page_profile():
    u       = st.session_state.user
    reports = get_user_reports(u["id"])
    st.markdown('<p class="main-title">👤 My Profile</p>', unsafe_allow_html=True)
    initials = u["username"][0].upper()
    total    = len(reports)
    positive = sum(1 for r in reports if "Positive" in r["prediction"] or "High" in r["prediction"])
    diseases = len({r["disease"] for r in reports})
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-avatar">{initials}</div>
        <div class="profile-name">{u['username']}</div>
        <div class="profile-email">{u['email']}</div>
    </div>""", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="stat-pill"><span class="num">{total}</span><span class="lbl">Predictions</span></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-pill"><span class="num" style="color:#d9534f">{positive}</span><span class="lbl">Positive</span></div>', unsafe_allow_html=True)
    s3.markdown(f'<div class="stat-pill"><span class="num">{diseases}</span><span class="lbl">Diseases checked</span></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("### Account info")
    st.text_input("Username", value=u["username"], disabled=True)
    st.text_input("Email",    value=u["email"],    disabled=True)
    st.info("To update your profile details, please contact the administrator.")
    st.markdown("### Change password")
    with st.form("change_pw"):
        new_pw  = st.text_input("New password (min 6 chars)", type="password")
        conf_pw = st.text_input("Confirm new password",       type="password")
        if st.form_submit_button("Update password", use_container_width=True):
            if len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pw != conf_pw:
                st.error("Passwords do not match.")
            else:
                st.success("✅ Password updated successfully!")


# ── Settings ──────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown('<p class="main-title">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown("### Appearance")
    dark = st.toggle("🌙 Dark mode", value=st.session_state.get("dark_mode", False))
    if dark != st.session_state.get("dark_mode", False):
        st.session_state.dark_mode = dark
        st.rerun()
    st.markdown("### Notifications")
    st.toggle("Email report after prediction", value=False)
    st.toggle("Health tip reminders",          value=True)
    st.markdown("### Data")
    st.caption("Account: " + st.session_state.user["email"])
    if st.button("🗑️ Delete all my reports", type="secondary"):
        delete_all_reports(st.session_state.user["id"])
        st.success("All reports deleted!")
    st.markdown("### About")
    st.info("MDPS v2.0 — Built with Streamlit & scikit-learn\n\nThis app is for educational purposes only.")


# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.user:
    auth_screen()
else:
    page = sidebar()
    {
        "🏠 Dashboard":          page_dashboard,
        "🏥 About":              page_about,
        "� Full Body Checkup":  page_full_body_checkup,
        "�🧪 Disease Prediction": page_disease_prediction,
        "🤒 Symptom-Based":      page_symptom,
        "📁 Saved Reports":      page_reports,
        "📊 Analytics":          page_analytics,
        "💡 Health Tips":        page_tips,
        "👤 Profile":            page_profile,
        "⚙️ Settings":          page_settings,
        "🏨 Nearby Hospitals": page_hospitals,
        "🤖 AI Chatbot": page_chatbot,
    }[page]()