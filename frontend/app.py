import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

API_BASE = "http://localhost:5000"

st.set_page_config(
    page_title="EventIQ v2 — ML Event Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── LANDING PAGE ─────────────────────────────────────────────── */
.landing-wrap {
    min-height: 88vh;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(37,99,235,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.10) 0%, transparent 60%),
        #060c18;
    border-radius: 20px; padding: 60px 40px 50px;
    position: relative; overflow: hidden;
}
.landing-wrap::before {
    content: ''; position: absolute; inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 32px 32px; pointer-events: none;
}
.landing-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4);
    color: #93c5fd; font-family: 'DM Mono', monospace;
    font-size: .75rem; letter-spacing: .12em;
    padding: 6px 18px; border-radius: 999px; margin-bottom: 28px;
}
.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.05; color: #f1f5f9; margin: 0 0 8px;
}
.landing-title span {
    background: linear-gradient(135deg, #3b82f6, #06b6d4, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.landing-sub {
    font-size: 1.15rem; color: rgba(203,213,225,.7);
    max-width: 560px; margin: 18px auto 40px; line-height: 1.7;
}
.stat-row {
    display: flex; gap: 20px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 44px;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 14px 24px;
    text-align: center; min-width: 110px;
}
.stat-pill .sv {
    font-family: 'Syne', sans-serif; font-size: 1.6rem;
    font-weight: 700; color: #f1f5f9; line-height: 1;
}
.stat-pill .sl {
    font-size: .72rem; color: rgba(148,163,184,.7);
    text-transform: uppercase; letter-spacing: .08em; margin-top: 4px;
}
.feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin-top: 10px;
}
.feat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 22px 20px; transition: all .25s;
}
.feat-card:hover {
    background: rgba(59,130,246,0.07);
    border-color: rgba(59,130,246,0.3);
    transform: translateY(-3px);
}
.feat-icon { font-size: 1.8rem; margin-bottom: 10px; }
.feat-title {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #f1f5f9; margin-bottom: 6px;
}
.feat-desc { font-size: .83rem; color: rgba(148,163,184,.8); line-height: 1.6; }
.tech-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    justify-content: center; margin-top: 8px;
}
.tech-tag {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 6px 14px;
    font-family: 'DM Mono', monospace;
    font-size: .78rem; color: rgba(148,163,184,.9);
}

/* ── METRIC CARDS ─────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(99,179,237,.2);
    border-radius: 12px; padding: 18px 20px;
    color: white; text-align: center;
}
.metric-card .val {
    font-size: 2rem; font-weight: 600; color: #63b3ed;
    line-height: 1.1; font-family: 'Syne', sans-serif;
}
.metric-card .lbl {
    font-size: .72rem; color: rgba(255,255,255,.5);
    margin-top: 3px; text-transform: uppercase; letter-spacing: .06em;
}

/* ── PREDICT BADGES ───────────────────────────────────────────── */
.attend-badge {
    background: linear-gradient(135deg, #065f46, #047857);
    color: #a7f3d0; font-size: 1.4rem; font-weight: 600;
    border-radius: 10px; padding: 14px 20px;
    text-align: center; border: 1px solid #059669;
}
.skip-badge {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fecaca; font-size: 1.4rem; font-weight: 600;
    border-radius: 10px; padding: 14px 20px;
    text-align: center; border: 1px solid #dc2626;
}

/* ── MISC ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-weight: 500; padding: .5rem 2rem; width: 100%;
}
.info-box {
    background: rgba(59,130,246,.08);
    border: 1px solid rgba(59,130,246,.25);
    border-radius: 8px; padding: 12px 16px;
    margin: 8px 0; font-size: .88rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_meta():
    try:
        r = requests.get(f"{API_BASE}/metadata", timeout=5)
        return r.json() if r.ok else None
    except:
        return None

@st.cache_data(ttl=10)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json() if r.ok else None
    except:
        return None

@st.cache_data(ttl=10)
def fetch_history():
    try:
        r = requests.get(f"{API_BASE}/history", timeout=5)
        return r.json() if r.ok else []
    except:
        return []

def online():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).ok
    except:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 EventIQ v2")
    st.markdown("*ML Event Analyst*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 Dashboard",
        "🔮 Predict",
        "📈 Insights",
        "📂 Analyze CSV",
        "🔍 EDA",
    ], label_visibility="collapsed")
    st.divider()
    st.caption(f"**API:** {'🟢 Online' if online() else '🔴 Offline'}")
    st.caption(f"Endpoint: `{API_BASE}`")
    meta = fetch_meta()
    if meta:
        st.divider()
        st.caption(f"Model: **{meta['model_name']}**")
        st.caption(f"Accuracy: **{meta['metrics']['accuracy']*100:.1f}%**")
        st.caption(f"AUC-ROC: **{meta['metrics']['auc']:.3f}**")
        st.caption(f"Dataset: **{meta['dataset_size']:,} rows**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME / LANDING
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    meta  = fetch_meta()
    acc   = f"{meta['metrics']['accuracy']*100:.1f}%" if meta else "88.0%"
    auc   = f"{meta['metrics']['auc']:.3f}"           if meta else "0.956"
    f1    = f"{meta['metrics']['f1']:.3f}"             if meta else "0.904"
    dsize = f"{meta['dataset_size']:,}"                if meta else "3,000"

    st.markdown(f"""
    <div class="landing-wrap">
        <div class="landing-badge">✦ AI &amp; DATA SCIENCE — AIDS260 PRACTICUM</div>
        <div class="landing-title">Event<span>IQ</span></div>
        <div class="landing-title" style="font-size:clamp(1.4rem,3vw,2.2rem);
            margin-top:-4px;
            background:linear-gradient(135deg,#94a3b8,#cbd5e1);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ML Event Analytics &amp; Participation Prediction
        </div>
        <div class="landing-sub">
            Predict who will attend your event before it happens.
            Powered by Gradient Boosting with
            <b style="color:#93c5fd">{acc} accuracy</b>
            trained on {dsize} event registration records.
        </div>
        <div class="stat-row">
            <div class="stat-pill">
                <div class="sv">{acc}</div><div class="sl">Accuracy</div>
            </div>
            <div class="stat-pill">
                <div class="sv">{auc}</div><div class="sl">AUC-ROC</div>
            </div>
            <div class="stat-pill">
                <div class="sv">{f1}</div><div class="sl">F1 Score</div>
            </div>
            <div class="stat-pill">
                <div class="sv">{dsize}</div><div class="sl">Training Rows</div>
            </div>
            <div class="stat-pill">
                <div class="sv">23</div><div class="sl">Features</div>
            </div>
            <div class="stat-pill">
                <div class="sv">6</div><div class="sl">API Endpoints</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("### What EventIQ Can Do")
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">🔮</div>
            <div class="feat-title">Attendance Prediction</div>
            <div class="feat-desc">Predict whether a registrant will attend
            using 23 engineered features including organizer score and social buzz.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📂</div>
            <div class="feat-title">Bulk CSV Analysis</div>
            <div class="feat-desc">Upload hundreds of registrations at once and
            get predicted attendance for every row with downloadable results.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📈</div>
            <div class="feat-title">Model Insights</div>
            <div class="feat-desc">Confusion matrix, classification report,
            feature importance treemap and probability distribution charts.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔍</div>
            <div class="feat-title">Exploratory Analysis</div>
            <div class="feat-desc">Upload any dataset and instantly explore
            distributions, correlations and attendance patterns across all features.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📊</div>
            <div class="feat-title">Live Dashboard</div>
            <div class="feat-desc">Real-time stats from prediction history —
            attendance rates by city, event type and probability distributions.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">⚡</div>
            <div class="feat-title">REST API</div>
            <div class="feat-desc">6 Flask endpoints for single prediction,
            batch analysis, model metadata, history and statistics.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛠 Tech Stack")
        st.markdown("""
        <div class="tech-row">
            <span class="tech-tag">Python 3.x</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Gradient Boosting</span>
            <span class="tech-tag">Flask</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Plotly</span>
            <span class="tech-tag">Pandas</span>
            <span class="tech-tag">NumPy</span>
            <span class="tech-tag">SQLite</span>
            <span class="tech-tag">Joblib</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 👥 Team")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:14px;padding:20px 22px;">
            <div style="margin-bottom:10px;">
                <span style="color:#93c5fd;font-family:'DM Mono',monospace;
                    font-size:.8rem;">01</span>
                <span style="color:#f1f5f9;font-weight:500;
                    margin-left:10px;">Ayush Aman</span>
                <span style="color:rgba(148,163,184,.6);font-size:.8rem;
                    margin-left:8px;">05315611924</span>
            </div>
            <div style="margin-bottom:10px;">
                <span style="color:#93c5fd;font-family:'DM Mono',monospace;
                    font-size:.8rem;">02</span>
                <span style="color:#f1f5f9;font-weight:500;
                    margin-left:10px;">Gautam Sharma</span>
                <span style="color:rgba(148,163,184,.6);font-size:.8rem;
                    margin-left:8px;">35215611924</span>
            </div>
            <div>
                <span style="color:#93c5fd;font-family:'DM Mono',monospace;
                    font-size:.8rem;">03</span>
                <span style="color:#f1f5f9;font-weight:500;
                    margin-left:10px;">Shubham Gupta</span>
                <span style="color:rgba(148,163,184,.6);font-size:.8rem;
                    margin-left:8px;">03415611924</span>
            </div>
            <div style="margin-top:14px;padding-top:14px;
                border-top:1px solid rgba(255,255,255,0.07);
                font-size:.82rem;color:rgba(148,163,184,.6);">
                Supervisor: <b style="color:#cbd5e1;">MR. YUG</b>
                &nbsp;|&nbsp; Sem 4th · Sec S13 &nbsp;|&nbsp; AIDS260
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ How It Works")
    for num, title, desc in [
        ("01", "Data Collection",
         "3,000 event registration records with 15 features including age, "
         "location, income, event rating, organizer score and social buzz."),
        ("02", "Feature Engineering",
         "7 interaction features created: rating×organizer, buzz×rating, "
         "loyalty score, income per km and more — 23 features total."),
        ("03", "Model Training",
         "Gradient Boosting Classifier with 300 estimators, depth 5, "
         "learning rate 0.05 — achieving 88% accuracy and 95.6% AUC-ROC."),
        ("04", "API Deployment",
         "Flask REST API serves predictions via 6 endpoints. "
         "Artifacts loaded once at startup for low-latency responses."),
        ("05", "Prediction",
         "Input details → preprocess → encode → scale → GBM inference "
         "→ probability + confidence returned instantly."),
    ]:
        st.markdown(f"""
        <div style="display:flex;gap:18px;align-items:flex-start;
            padding:16px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="font-family:'DM Mono',monospace;font-size:1.1rem;
                color:#3b82f6;font-weight:500;min-width:32px;">{num}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                    color:#f1f5f9;margin-bottom:4px;">{title}</div>
                <div style="font-size:.87rem;color:rgba(148,163,184,.75);
                    line-height:1.6;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;font-size:.8rem;
        color:rgba(100,116,139,.6);padding:28px 0 8px;">
        EventIQ v2 &nbsp;·&nbsp; AIDS260 Practicum &nbsp;·&nbsp;
        Dr. Akhilesh Das Gupta Institute of Professional Studies
        &nbsp;·&nbsp; Even Session 2024-25
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("Event Participation Dashboard")
    st.caption("Live model metrics and prediction analytics")

    meta  = fetch_meta()
    stats = fetch_stats()

    if meta:
        m = meta['metrics']
        cols = st.columns(4)
        for col, (val, lbl) in zip(cols, [
            (f"{m['accuracy']*100:.1f}%", "Test Accuracy"),
            (f"{m['auc']:.3f}",            "AUC-ROC"),
            (f"{m['f1']:.3f}",             "F1 Score"),
            (f"{m['cv_acc']*100:.1f}%",    "CV Accuracy"),
        ]):
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            f'<div class="info-box">🤖 <b>Model:</b> {meta["model_name"]}'
            f' &nbsp;|&nbsp; <b>Features:</b> {meta["n_features"]}'
            f' &nbsp;|&nbsp; <b>Rows:</b> {meta["dataset_size"]:,}'
            f' &nbsp;|&nbsp; <b>New signals:</b> organizer_score, social_buzz</div>',
            unsafe_allow_html=True)

    if stats and stats['total_predictions'] > 0:
        st.divider()
        st.subheader("Prediction History")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions",   stats['total_predictions'])
        c2.metric("Predicted Attending", stats['total_attending'])
        c3.metric("Predicted Skipping",  stats['total_skipping'])
        c4.metric("Attendance Rate",     f"{stats['attendance_rate']}%")

        col1, col2 = st.columns(2)
        if stats.get('by_event_type'):
            df_t = pd.DataFrame(stats['by_event_type'])
            fig  = px.bar(df_t, x='event_type', y=['attending', 'total'],
                          title="Predictions by Event Type",
                          barmode='overlay', opacity=0.85,
                          color_discrete_sequence=['#3b82f6', '#1e3a5f'],
                          template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               legend_title_text='')
            col1.plotly_chart(fig, use_container_width=True)

        if stats.get('by_location'):
            df_l = pd.DataFrame(stats['by_location'])
            df_l['rate'] = (df_l['attending'] / df_l['total'] * 100).round(1)
            fig2 = px.bar(df_l, x='location', y='rate',
                          title="Attendance Rate by City (%)",
                          color='rate', color_continuous_scale='Blues',
                          template='plotly_dark')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                coloraxis_showscale=False)
            col2.plotly_chart(fig2, use_container_width=True)

    if meta and meta.get('importances'):
        st.subheader("Feature Importance (Gradient Boosting)")
        imp    = meta['importances']
        df_imp = pd.DataFrame({
            'Feature':    list(imp.keys()),
            'Importance': list(imp.values())
        }).sort_values('Importance')
        fig3 = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues',
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    if not stats or stats['total_predictions'] == 0:
        st.info("No predictions yet — head to 🔮 Predict to get started!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("Predict Event Attendance")
    st.caption("Fill in registrant and event details to get an ML-powered prediction")

    meta = fetch_meta()
    lc   = meta['label_classes'] if meta else {
        'gender':     ['Female', 'Male', 'Other'],
        'location':   ['Bangalore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai', 'Pune'],
        'event_type': ['Art', 'Business', 'Education', 'Music', 'Sports', 'Tech'],
    }

    with st.form("predict_form"):
        st.markdown("#### 👤 Registrant Profile")
        c1, c2, c3 = st.columns(3)
        age    = c1.number_input("Age", 18, 80, 28)
        gender = c2.selectbox("Gender",   lc['gender'])
        loc    = c3.selectbox("City",     lc['location'])

        st.markdown("#### 🎪 Event Details")
        c4, c5 = st.columns(2)
        event_type   = c4.selectbox("Event Type", lc['event_type'])
        event_rating = c5.slider("Event Rating ⭐", 1.0, 5.0, 4.0, 0.1)

        c6, c7 = st.columns(2)
        reg_date   = c6.date_input("Registration Date",
                                    value=date.today() - timedelta(days=14))
        event_date = c7.date_input("Event Date",
                                    value=date.today() + timedelta(days=7))

        st.markdown("#### 📊 Organizer & Buzz")
        c8, c9 = st.columns(2)
        organizer_score = c8.slider(
            "Organizer Score 🏆", 1.0, 5.0, 3.5, 0.1,
            help="Past reputation of the event organizer (1=poor, 5=excellent)")
        social_buzz = c9.slider(
            "Social Buzz 📱", 0, 100, 50,
            help="Online engagement score (0=none, 100=viral)")

        st.markdown("#### 💰 Socioeconomic")
        c10, c11, c12 = st.columns(3)
        income          = c10.number_input("Income (₹/yr)", 10000, 200000, 50000, 1000)
        distance_km     = c11.number_input("Distance (km)", 1, 200, 15)
        previous_events = c12.number_input("Past Events Attended", 0, 50, 3)

        submitted = st.form_submit_button("🔮 Predict Attendance")

    if submitted:
        payload = {
            'age': age, 'gender': gender, 'location': loc,
            'event_type': event_type, 'event_rating': event_rating,
            'organizer_score': organizer_score, 'social_buzz': social_buzz,
            'registration_date': str(reg_date), 'event_date': str(event_date),
            'income': income, 'distance_km': distance_km,
            'previous_events': previous_events,
        }
        with st.spinner("Running Gradient Boosting prediction..."):
            try:
                r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                if r.ok:
                    res  = r.json()
                    pred = res['prediction']
                    prob = res['probability']

                    badge = "attend-badge" if pred == 1 else "skip-badge"
                    icon  = "✅" if pred == 1 else "❌"
                    st.markdown(
                        f'<div class="{badge}">{icon} {res["label"]}'
                        f' &nbsp;|&nbsp; Confidence: {res["confidence"]}</div>',
                        unsafe_allow_html=True)
                    st.markdown("")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100,
                        title={"text": "Attendance Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#3b82f6"},
                            "steps": [
                                {"range": [0,  40], "color": "#7f1d1d"},
                                {"range": [40, 60], "color": "#78350f"},
                                {"range": [60, 100],"color": "#064e3b"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 2},
                                "value": 50
                            }
                        }
                    ))
                    fig.update_layout(height=280,
                                       paper_bgcolor='rgba(0,0,0,0)',
                                       font_color='white')
                    st.plotly_chart(fig, use_container_width=True)

                    # Driver bar chart
                    st.subheader("Prediction Drivers")
                    drivers = {
                        "Event Rating":     (event_rating - 2.5) / 2.5,
                        "Organizer Score":  (organizer_score - 1) / 4,
                        "Social Buzz":       social_buzz / 100,
                        "Proximity":         1 - (distance_km / 200),
                        "Past Attendance":   previous_events / 10,
                    }
                    df_d = pd.DataFrame({
                        'Driver': list(drivers.keys()),
                        'Score':  list(drivers.values())
                    })
                    fig2 = px.bar(df_d, x='Score', y='Driver', orientation='h',
                                   color='Score',
                                   color_continuous_scale='RdYlGn',
                                   range_x=[0, 1], template='plotly_dark',
                                   title="Input Signal Strength (0=weak, 1=strong)")
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True)

                else:
                    st.error(f"API error: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach Flask API. Make sure it is running on port 5000.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Insights":
    st.title("Model Insights")
    meta = fetch_meta()
    if not meta:
        st.error("Cannot reach Flask API.")
        st.stop()

    c1, c2 = st.columns(2)

    # Confusion matrix
    cm  = meta['confusion_matrix']
    fig = px.imshow(cm, text_auto=True,
                    x=['Predicted: No', 'Predicted: Yes'],
                    y=['Actual: No',    'Actual: Yes'],
                    color_continuous_scale='Blues',
                    template='plotly_dark',
                    title='Confusion Matrix (Test Set)')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                       coloraxis_showscale=False)
    c1.plotly_chart(fig, use_container_width=True)

    # Radar chart
    m    = meta['metrics']
    fig2 = go.Figure(go.Scatterpolar(
        r=[m['accuracy'], m['auc'], m['f1'], m['cv_acc'], m['accuracy']],
        theta=['Accuracy', 'AUC-ROC', 'F1', 'CV Acc', 'Accuracy'],
        fill='toself', line_color='#3b82f6',
        fillcolor='rgba(59,130,246,0.2)'
    ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        title='Model Performance Radar',
        paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    c2.plotly_chart(fig2, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    cr   = meta['class_report']
    rows = [{'Class': lbl, **{k: round(v, 3) for k, v in vals.items()}}
            for lbl, vals in cr.items() if isinstance(vals, dict)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Treemap
    st.subheader("Feature Importance Treemap")
    imp    = meta['importances']
    df_imp = pd.DataFrame({
        'Feature':    list(imp.keys()),
        'Importance': list(imp.values())
    })
    fig3 = px.treemap(df_imp, path=['Feature'], values='Importance',
                       color='Importance', color_continuous_scale='Blues',
                       template='plotly_dark')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    df_imp['Importance %'] = (df_imp['Importance'] * 100).round(2)
    st.dataframe(
        df_imp.sort_values('Importance', ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True)

    # History distribution
    history = fetch_history()
    if history:
        st.subheader("Prediction Probability Distribution")
        df_h = pd.DataFrame(history)
        df_h['label'] = df_h['prediction'].map({1: 'Attending', 0: 'Not Attending'})
        fig4 = px.histogram(df_h, x='probability', color='label',
                             nbins=20, barmode='overlay', opacity=0.75,
                             color_discrete_map={
                                 'Attending':     '#3b82f6',
                                 'Not Attending': '#ef4444'},
                             template='plotly_dark',
                             title='Distribution of Predicted Probabilities')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYZE CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Analyze CSV":
    st.title("Bulk CSV Analysis")
    st.caption("Upload a CSV of registrations for batch attendance predictions")
    st.info(
        "**Required columns:** age, gender, location, event_type, previous_events, "
        "income, event_rating, distance_km, **organizer_score**, **social_buzz**, "
        "registration_date, event_date")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_prev = pd.read_csv(uploaded)
        uploaded.seek(0)
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_prev.head(), use_container_width=True, hide_index=True)

        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Analysing all rows..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/analyze",
                        files={'file': ('data.csv', uploaded, 'text/csv')},
                        timeout=300)
                    if r.ok:
                        res    = r.json()
                        attend = res['predicted_attend']
                        skip   = res['predicted_skip']
                        total  = res['total']

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Rows",           total)
                        c2.metric("Predicted Attending",  attend)
                        c3.metric("Predicted Skipping",   skip)
                        c4.metric("Attendance Rate",      f"{res['attendance_rate']}%")

                        c5, c6 = st.columns(2)
                        fig = px.pie(
                            values=[attend, skip],
                            names=['Will Attend', 'Will Not Attend'],
                            hole=0.55,
                            color_discrete_sequence=['#3b82f6', '#ef4444'],
                            template='plotly_dark',
                            title='Predicted Attendance Split')
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                        c5.plotly_chart(fig, use_container_width=True)

                        df_res = pd.DataFrame(res['rows'])
                        if 'event_type' in df_res.columns:
                            df_et = (df_res.groupby('event_type')['prediction']
                                     .mean().reset_index())
                            df_et.columns = ['event_type', 'attend_rate']
                            df_et['attend_rate'] = (
                                df_et['attend_rate'] * 100).round(1)
                            fig2 = px.bar(df_et, x='event_type', y='attend_rate',
                                           color='attend_rate',
                                           color_continuous_scale='Blues',
                                           template='plotly_dark',
                                           title='Attendance Rate by Event Type (%)')
                            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                                coloraxis_showscale=False)
                            c6.plotly_chart(fig2, use_container_width=True)

                        st.subheader("Full Results")
                        st.dataframe(df_res, use_container_width=True,
                                      hide_index=True)
                        st.download_button(
                            "⬇️ Download Results CSV",
                            df_res.to_csv(index=False).encode(),
                            "predictions_output.csv", "text/csv")
                    else:
                        st.error(f"API error: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach Flask API.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    st.caption("Upload your dataset to explore distributions and attendance patterns")

    uploaded = st.file_uploader("Upload dataset CSV", type=['csv'], key='eda')
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader(
            f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)

        # Attendance distribution
        att_counts = df['attended'].value_counts().reset_index()
        att_counts.columns = ['Status', 'Count']
        att_counts['Status'] = att_counts['Status'].map(
            {1: 'Attended', 0: 'Not Attended'})
        fig = px.pie(att_counts, values='Count', names='Status', hole=0.5,
                      color_discrete_sequence=['#3b82f6', '#ef4444'],
                      template='plotly_dark',
                      title='Overall Attendance Distribution')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        c1.plotly_chart(fig, use_container_width=True)

        if 'event_type' in df.columns:
            df_et = df.groupby('event_type')['attended'].mean().reset_index()
            df_et.columns = ['Event Type', 'Attendance Rate']
            df_et['Attendance Rate'] = (df_et['Attendance Rate'] * 100).round(1)
            fig2 = px.bar(df_et, x='Event Type', y='Attendance Rate',
                           color='Attendance Rate',
                           color_continuous_scale='Blues',
                           template='plotly_dark',
                           title='Attendance Rate by Event Type (%)')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                coloraxis_showscale=False)
            c2.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        if 'event_rating' in df.columns:
            fig3 = px.histogram(
                df, x='event_rating',
                color=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                nbins=25, barmode='overlay', opacity=0.75,
                color_discrete_map={
                    'Attended': '#3b82f6', 'Not Attended': '#ef4444'},
                template='plotly_dark',
                title='Event Rating Distribution by Attendance')
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                legend_title_text='')
            c3.plotly_chart(fig3, use_container_width=True)

        if 'distance_km' in df.columns:
            fig4 = px.histogram(
                df, x='distance_km',
                color=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                nbins=25, barmode='overlay', opacity=0.75,
                color_discrete_map={
                    'Attended': '#3b82f6', 'Not Attended': '#ef4444'},
                template='plotly_dark',
                title='Distance Distribution by Attendance')
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                legend_title_text='')
            c4.plotly_chart(fig4, use_container_width=True)

        # New features
        if 'organizer_score' in df.columns and 'social_buzz' in df.columns:
            st.subheader("New Feature Analysis")
            c5, c6 = st.columns(2)
            fig5 = px.box(
                df,
                x=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                y='organizer_score',
                color=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                color_discrete_map={
                    'Attended': '#3b82f6', 'Not Attended': '#ef4444'},
                template='plotly_dark',
                title='Organizer Score vs Attendance')
            fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False)
            c5.plotly_chart(fig5, use_container_width=True)

            fig6 = px.box(
                df,
                x=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                y='social_buzz',
                color=df['attended'].map({1: 'Attended', 0: 'Not Attended'}),
                color_discrete_map={
                    'Attended': '#3b82f6', 'Not Attended': '#ef4444'},
                template='plotly_dark',
                title='Social Buzz vs Attendance')
            fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False)
            c6.plotly_chart(fig6, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include='number')
        corr   = num_df.corr()
        fig7   = px.imshow(corr, text_auto='.2f', aspect='auto',
                            color_continuous_scale='RdBu_r',
                            template='plotly_dark',
                            title='Feature Correlation Matrix')
        fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig7, use_container_width=True)

    else:
        st.info("Upload `event_dataset_3000.csv` from the data/ folder to explore.")