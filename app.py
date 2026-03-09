import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Calorie Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #f8fafc; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #38b2ac 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .hero p  { font-size: 1rem; opacity: 0.85; margin: 0; }

    /* Cards */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
        border: 1px solid #e8edf3;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #e8f4fd;
        padding-bottom: 0.7rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 14px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(30,58,95,0.3);
    }
    .result-box .label { font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.4rem; }
    .result-box .calories { font-size: 3.5rem; font-weight: 700; line-height: 1; }
    .result-box .unit { font-size: 1rem; opacity: 0.75; }
    .result-box .badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Nutrient pill */
    .nutrient-pill {
        display: inline-block;
        background: #e8f4fd;
        color: #1e3a5f;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f4f8;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Slider label */
    .slider-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: -12px;
    }

    /* Info metric */
    .info-metric {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .info-metric .val { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; }
    .info-metric .lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.9; }

    .stSelectbox label, .stNumberInput label, .stSlider label { font-weight: 500; color: #374151; font-size: 0.9rem; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    # Try loading from pickle first
    if os.path.exists("calorie_model.pkl"):
        model = pickle.load(open("calorie_model.pkl", "rb"))
        df = pd.read_csv("food.csv") if os.path.exists("food.csv") else None
        return model, df

    # Otherwise train fresh
    if not os.path.exists("food.csv"):
        return None, None

    df = pd.read_csv("food.csv")
    features = ['Data.Protein', 'Data.Carbohydrate', 'Data.Fiber',
                 'Data.Sugar Total', 'Data.Fat.Total Lipid']
    x = df[features].fillna(0)
    y = df['Data.Kilocalories'].fillna(0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model, df

model, df = load_model_and_data()


def get_calorie_label(cal):
    if cal < 50:   return ("Very Low Calorie", "#10b981")
    if cal < 150:  return ("Low Calorie", "#34d399")
    if cal < 300:  return ("Moderate Calorie", "#f59e0b")
    if cal < 500:  return ("High Calorie", "#f97316")
    return ("Very High Calorie", "#ef4444")


def predict(protein, carbs, fiber, sugar, fat):
    inp = np.array([[protein, carbs, fiber, sugar, fat]])
    return model.predict(inp)[0]


# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🥗 Food Calorie Predictor</h1>
    <p>Search by food name or enter nutritional values manually to instantly estimate calories</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Could not load model or data. Place **food.csv** and/or **calorie_model.pkl** in the same directory as this app.")
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Search by Food Name", "✏️  Enter Nutrients Manually"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Search by Name
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if df is None:
        st.warning("food.csv not found. Please use the manual entry tab.")
    else:
        col_search, col_gap = st.columns([3, 1])
        with col_search:
            search_query = st.text_input(
                "Type a food name",
                placeholder="e.g. cheese, chicken, butter, milk…",
                label_visibility="collapsed"
            )

        if search_query:
            mask = df['Description'].str.contains(search_query, case=False, na=False)
            results = df[mask].reset_index(drop=True)

            if results.empty:
                st.info(f"No food items found matching **'{search_query}'**. Try a different keyword.")
            else:
                st.markdown(f"<p style='color:#64748b; font-size:0.88rem;'>Found <b>{len(results)}</b> result(s)</p>", unsafe_allow_html=True)

                options = results['Description'].tolist()
                selected = st.selectbox("Select a food item", options)

                row = results[results['Description'] == selected].iloc[0]
                protein  = row.get('Data.Protein', 0)
                carbs    = row.get('Data.Carbohydrate', 0)
                fiber    = row.get('Data.Fiber', 0)
                sugar    = row.get('Data.Sugar Total', 0)
                fat      = row.get('Data.Fat.Total Lipid', 0)
                actual   = row.get('Data.Kilocalories', None)

                predicted = predict(protein, carbs, fiber, sugar, fat)
                label, color = get_calorie_label(predicted)

                col_res, col_info = st.columns([1, 1.6])

                with col_res:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="label">Predicted Calories (per 100g)</div>
                        <div class="calories">{predicted:.0f}</div>
                        <div class="unit">kcal</div>
                        <span class="badge" style="background:rgba(255,255,255,0.2)">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_info:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">📊 Nutritional Breakdown</div>', unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    metrics = [
                        ("Protein", protein, "g"),
                        ("Carbs", carbs, "g"),
                        ("Fat", fat, "g"),
                        ("Fiber", fiber, "g"),
                        ("Sugar", sugar, "g"),
                    ]
                    if actual:
                        metrics.append(("Actual kcal", actual, ""))

                    cols = st.columns(3)
                    for i, (name, val, unit) in enumerate(metrics):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div class="info-metric">
                                <div class="val">{val:.1f}<span style='font-size:0.7rem;color:#94a3b8'>{unit}</span></div>
                                <div class="lbl">{name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if actual:
                        diff = abs(predicted - actual)
                        acc = max(0, 100 - (diff / max(actual, 1) * 100))
                        st.markdown(f"""
                        <div style="margin-top:1rem; padding:0.7rem 1rem; background:#f0fdf4; border-radius:8px; border-left:3px solid #10b981;">
                            <span style="font-size:0.85rem; color:#065f46;">
                                ✅ Model accuracy for this item: <b>{acc:.1f}%</b>
                                &nbsp;|&nbsp; Actual: <b>{actual:.0f} kcal</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual Entry
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Enter Nutrient Values (per 100g)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        protein = st.number_input("🥩 Protein (g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        carbs   = st.number_input("🍞 Carbohydrates (g)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        fiber   = st.number_input("🌾 Fiber (g)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    with col2:
        sugar   = st.number_input("🍬 Sugar Total (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        fat     = st.number_input("🧈 Fat / Total Lipid (g)", min_value=0.0, max_value=100.0, value=8.0, step=0.1)

        macro_total = protein + carbs + fat
        if macro_total > 100:
            st.warning(f"⚠️ Protein + Carbs + Fat = {macro_total:.1f}g — typically ≤ 100g per 100g food.")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Predict Calories"):
        predicted = predict(protein, carbs, fiber, sugar, fat)
        label, color = get_calorie_label(predicted)

        col_r, col_b = st.columns([1, 1.6])

        with col_r:
            st.markdown(f"""
            <div class="result-box">
                <div class="label">Estimated Calories (per 100g)</div>
                <div class="calories">{predicted:.0f}</div>
                <div class="unit">kcal</div>
                <span class="badge" style="background:rgba(255,255,255,0.2)">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📈 Calorie Contribution by Macro</div>', unsafe_allow_html=True)

            protein_cal = protein * 4
            carbs_cal   = carbs * 4
            fat_cal     = fat * 9
            total_macro_cal = protein_cal + carbs_cal + fat_cal

            if total_macro_cal > 0:
                for name, cal, color_bar in [
                    ("Protein", protein_cal, "#3b82f6"),
                    ("Carbohydrates", carbs_cal, "#f59e0b"),
                    ("Fat", fat_cal, "#ef4444"),
                ]:
                    pct = cal / total_macro_cal * 100
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex; justify-content:space-between; font-size:0.83rem; color:#374151; margin-bottom:3px;">
                            <span><b>{name}</b></span>
                            <span>{cal:.0f} kcal &nbsp;({pct:.0f}%)</span>
                        </div>
                        <div style="background:#e2e8f0; border-radius:999px; height:10px;">
                            <div style="background:{color_bar}; width:{pct}%; height:10px; border-radius:999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#94a3b8; font-size:0.8rem;">
    Powered by a Random Forest Regressor trained on USDA Nutrient Data &nbsp;•&nbsp;
    Predictions are per 100g serving
</div>
""", unsafe_allow_html=True)