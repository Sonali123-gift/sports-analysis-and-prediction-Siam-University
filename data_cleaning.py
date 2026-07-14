"""
Sports Participation Survey — Data Cleaning App
================================================
Run with:  streamlit run sports_cleaning_app.py
Requires:  pip install streamlit pandas numpy openpyxl plotly
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Survey Cleaner",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSlider label { color: #c9d1d9 !important; }

/* Main background */
.main { background: #f6f8fa; }

/* Title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0d1117;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #57606a;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: white;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #57606a;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0d1117;
}
.metric-card .delta {
    font-size: 0.78rem;
    color: #1a7f37;
    font-weight: 500;
}
.metric-card .delta.red { color: #cf222e; }

/* Step badges */
.step-badge {
    display: inline-block;
    background: #0d1117;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
.step-title {
    font-weight: 600;
    font-size: 1rem;
    color: #0d1117;
    margin-bottom: 0.2rem;
}
.step-detail {
    font-size: 0.88rem;
    color: #57606a;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #57606a;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* Download button */
.stDownloadButton > button {
    background: #0d1117 !important;
    color: #f0f6fc !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
}
.stDownloadButton > button:hover {
    background: #161b22 !important;
    box-shadow: 0 0 0 2px #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
SPORT_MAP = {
    "Esport":                             "E-Sports",
    "Basktballandvolleyball":             "Basketball / Volleyball",
    "Combat Sporr":                       "Combat Sports",
    "Mma,Boxing,Bjj, Aikido ,Taekwondo": "Combat Sports",
    "Table Tennis, Basketball":           "Table Tennis / Basketball",
    "Javelin, Relay And Sprinting":       "Athletics",
    "Motogp/F1":                          "Motorsport",
}
SPORTS_HRS_MAP = {"0": 0, "1–2": 1.5, "3–5": 4.0, "6–8": 7.0, "All Day": 12.0, "30+": 30.0}
SLEEP_MAP      = {"Less Than 5": 4.0, "5–6": 5.5, "6–7": 6.5, "7–8": 7.5, "8+": 9.0}

RENAME = {
    "Preferred Sport":                                       "Sport",
    "Weekly Sports Participation Hours ":                    "SportsHrs",
    "Sleep Hours per Day":                                   "Sleep",
    " Stress Level":                                         "Stress",
    "Starting Age of Sports":                                "StartAge",
    "What is your height? ":                                 "Height",
    "How would you describe your physical activity level?":  "ActivityLevel",
    "Reason for Playing Sports":                             "Reason",
    "What prevents you from participating in sports?":       "Barrier",
    "Do You Want More Sports Facilities?":                   "MoreFacilities",
    "Academic Workload":                                     "AcademicWorkload",
    "Peer Influence":                                        "PeerInfluence",
    "University / Institution":                              "University",
}
CAT_COLS = ["Gender","Nationality","Faculty","University","ActivityLevel",
            "Reason","Barrier","MoreFacilities","AcademicWorkload","PeerInfluence"]
NUM_COLS = ["SportsHrs_num","Sleep_num","Stress_num","Height_num"]


# ── Sidebar: controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cleaning Controls")
    st.markdown("---")

    step1 = st.checkbox("Remove null/empty Sport rows",    value=True)
    step2 = st.checkbox("Standardize Sport text",          value=True)

    min_freq = st.slider("Min participants per sport (Step 3)", 1, 10, 2)

    step4 = st.checkbox("Fill categorical NaN with mode",  value=True)
    step5 = st.checkbox("Encode ordinal text → numeric",   value=True)

    impute = st.selectbox("Numeric NaN imputation (Step 6)", ["Median","Mean"])

    iqr_mult = st.slider("IQR multiplier (Step 7)", 1.0, 3.0, 1.5, 0.1)
    step7 = st.checkbox("Remove IQR outliers",             value=True)

    st.markdown("---")
    uploaded = st.file_uploader("Upload your own dataset (.xlsx)", type=["xlsx"])
    st.caption("Leave blank to use the built-in demo file.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏅 Sports Survey<br>Data Cleaner</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Interactive pipeline · IQR outlier removal · Downloadable output</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_default():
    return pd.read_excel("Sports_Participation_Survey_Group_4_Responses__2_.xlsx")
if uploaded:
    try:
        raw = pd.read_excel(uploaded)
        st.success(f"✅ Uploaded file loaded — {raw.shape[0]} rows × {raw.shape[1]} cols")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    st.info("👆 Please upload an Excel (.xlsx) file from the sidebar to begin.")
    st.stop()


# ── Pipeline ──────────────────────────────────────────────────────────────────
log = []   # (step_label, action, detail, rows_after)

df = raw.copy()
df.rename(columns=RENAME, inplace=True)
log.append(("0 — Raw Data", "Load dataset",
             f"{df.shape[0]} rows × {df.shape[1]} columns", df.shape[0]))

# Step 1
if step1:
    before = len(df)
    df["Sport"] = df["Sport"].astype(str).str.strip()
    df = df[~df["Sport"].isin(["nan","","-","NaN"])].copy()
    removed = before - len(df)
    log.append(("1 — Null Sport", "Remove null/empty rows",
                 f"{removed} row(s) dropped where Sport was NaN / empty / '-'", len(df)))

# Step 2
if step2:
    def std_sport(v):
        v = str(v).strip().title()
        return SPORT_MAP.get(v, v)
    df["Sport"] = df["Sport"].apply(std_sport)
    log.append(("2 — Standardize", "Title-case + fix typos",
                 f"Applied {len(SPORT_MAP)} mapping rules; 'football'→'Football' etc.", len(df)))

# Step 3
before = len(df)
counts = df["Sport"].value_counts()
valid  = counts[counts >= min_freq].index
df = df[df["Sport"].isin(valid)].copy()
removed = before - len(df)
log.append(("3 — Low-freq", f"Min freq = {min_freq}",
             f"{removed} rows removed; {len(valid)} sports kept", len(df)))

# Step 4
if step4:
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    filled_cols = []
    for col in CAT_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])
                filled_cols.append(col)
    log.append(("4 — Categoricals", "Mode imputation + trim",
                 f"Trimmed all cat cols; mode-filled: {filled_cols or 'none'}", len(df)))

# Step 5
if step5:
    df["SportsHrs_num"] = df["SportsHrs"].astype(str).str.strip().str.title().map(SPORTS_HRS_MAP)
    df["Sleep_num"]     = df["Sleep"].astype(str).str.strip().str.title().map(SLEEP_MAP)
    df["Stress_num"]    = pd.to_numeric(df["Stress"], errors="coerce")
    df["Height_num"]    = pd.to_numeric(df["Height"], errors="coerce")
    # Replace 0 values in SportsHrs_num with column mean (excluding zeros)
    nonzero_mean = df.loc[df["SportsHrs_num"] > 0, "SportsHrs_num"].mean()
    zero_count   = (df["SportsHrs_num"] == 0).sum()
    df["SportsHrs_num"] = df["SportsHrs_num"].replace(0, round(nonzero_mean, 2))
    log.append(("5 — Encode", "Text ranges → numeric midpoints + zero fix",
                 f"SportsHrs & Sleep mapped; {zero_count} zero SportsHrs value(s) → mean ({nonzero_mean:.2f} hrs)", len(df)))

    # Step 6 — imputation
    available_num = [c for c in NUM_COLS if c in df.columns]
    imputed = []
    for col in available_num:
        n = df[col].isnull().sum()
        if n:
            fill_val = df[col].median() if impute == "Median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            imputed.append(f"{col} ({n} NaN → {fill_val:.1f})")
    log.append(("6 — Impute NaN", f"{impute} imputation",
                 "; ".join(imputed) if imputed else "No NaN found in numeric columns", len(df)))

# Step 7
if step7 and step5:
    before = len(df)
    outlier_detail = []
    available_num = [c for c in NUM_COLS if c in df.columns]
    for col in available_num:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            outlier_detail.append(f"{col}: {n_out} removed [{lo:.1f}–{hi:.1f}]")
        df = df[(df[col] >= lo) & (df[col] <= hi)].copy()
    removed = before - len(df)
    log.append(("7 — Outliers", f"IQR × {iqr_mult}",
                 "; ".join(outlier_detail) if outlier_detail else "No outliers found",
                 len(df)))

df_clean = df.reset_index(drop=True)


# ── Top metrics ───────────────────────────────────────────────────────────────
raw_n   = raw.shape[0]
clean_n = df_clean.shape[0]
removed_total = raw_n - clean_n
pct_kept = clean_n / raw_n * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw Rows",     f"{raw_n:,}")
m2.metric("Cleaned Rows", f"{clean_n:,}", delta=f"-{removed_total} removed")
m3.metric("Sports Kept",  df_clean["Sport"].nunique() if "Sport" in df_clean.columns else "—")
m4.metric("Data Retained", f"{pct_kept:.1f}%")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Cleaning Log", "📊 Visuals", "🗃️ Cleaned Data"])


# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Step-by-step cleaning record</div>', unsafe_allow_html=True)

    for step_label, action, detail, rows_after in log:
        with st.container():
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f'<div class="step-badge">{step_label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="step-title">{action}</div>', unsafe_allow_html=True)
                st.caption(f"→ {rows_after:,} rows")
            with col_b:
                st.markdown(f'<div class="step-detail">{detail}</div>', unsafe_allow_html=True)
            st.divider()


# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if "Sport" not in df_clean.columns:
        st.info("Run Steps 1–3 to see sport distribution.")
    else:
        SPORT_PALETTE = ["#6366F1","#F59E0B","#10B981","#EF4444","#3B82F6","#EC4899","#14B8A6","#F97316"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">Sport Distribution</div>', unsafe_allow_html=True)
            sport_df = df_clean["Sport"].value_counts().reset_index()
            sport_df.columns = ["Sport", "Count"]
            fig1 = px.bar(
                sport_df, x="Count", y="Sport", orientation="h",
                color="Sport",
                color_discrete_sequence=SPORT_PALETTE,
                template="plotly_white",
            )
            fig1.update_layout(
                margin=dict(l=0,r=0,t=10,b=0),
                showlegend=False,
                font_family="DM Sans",
                yaxis=dict(categoryorder="total ascending"),
                height=320,
            )
            fig1.update_traces(marker_line_width=0)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Gender Breakdown</div>', unsafe_allow_html=True)
            if "Gender" in df_clean.columns:
                gender_df = df_clean["Gender"].value_counts().reset_index()
                gender_df.columns = ["Gender","Count"]
                fig2 = px.pie(
                    gender_df, names="Gender", values="Count",
                    color_discrete_sequence=["#6366F1","#F59E0B","#10B981","#EF4444"],
                    template="plotly_white",
                    hole=0.45,
                )
                fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), font_family="DM Sans", height=320)
                st.plotly_chart(fig2, use_container_width=True)

        if step5 and "SportsHrs_num" in df_clean.columns:
            st.markdown('<div class="section-header">Numeric Distributions (post-cleaning)</div>', unsafe_allow_html=True)
            available_num = [c for c in NUM_COLS if c in df_clean.columns]
            hist_colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"]
            labels = {
                "SportsHrs_num": "Sports Hrs/Week",
                "Sleep_num":     "Sleep Hrs/Day",
                "Stress_num":    "Stress Level",
                "Height_num":    "Height (cm)",
            }
            cols = st.columns(len(available_num))
            for i, col in enumerate(available_num):
                with cols[i]:
                    fig = px.histogram(
                        df_clean, x=col, nbins=12,
                        labels={col: labels.get(col, col)},
                        color_discrete_sequence=[hist_colors[i % len(hist_colors)]],
                        template="plotly_white",
                    )
                    fig.update_layout(
                        title=dict(text=labels.get(col, col), font=dict(size=13)),
                        margin=dict(l=0,r=0,t=36,b=0),
                        font_family="DM Sans",
                        height=240,
                        bargap=0.05,
                        showlegend=False,
                    )
                    fig.update_traces(marker_line_color="white", marker_line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

        if step5 and "Sport" in df_clean.columns and "Stress_num" in df_clean.columns:
            st.markdown('<div class="section-header">Stress Level by Sport</div>', unsafe_allow_html=True)
            fig3 = px.box(
                df_clean, x="Sport", y="Stress_num",
                color="Sport",
                color_discrete_sequence=SPORT_PALETTE,
                template="plotly_white",
                labels={"Stress_num": "Stress Level", "Sport": ""},
            )
            fig3.update_layout(
                showlegend=False, font_family="DM Sans",
                margin=dict(l=0,r=0,t=10,b=0), height=340,
            )
            st.plotly_chart(fig3, use_container_width=True)

        if step5 and "Sport" in df_clean.columns and "SportsHrs_num" in df_clean.columns:
            st.markdown('<div class="section-header">Avg Sports Hours by Sport</div>', unsafe_allow_html=True)
            avg_hrs = df_clean.groupby("Sport")["SportsHrs_num"].mean().reset_index()
            avg_hrs.columns = ["Sport", "Avg Hrs"]
            avg_hrs = avg_hrs.sort_values("Avg Hrs", ascending=False)
            fig4 = px.bar(
                avg_hrs, x="Sport", y="Avg Hrs",
                color="Sport",
                color_discrete_sequence=SPORT_PALETTE,
                template="plotly_white",
                labels={"Avg Hrs": "Avg Hours/Week", "Sport": ""},
                text=avg_hrs["Avg Hrs"].round(1),
            )
            fig4.update_traces(textposition="outside", marker_line_width=0)
            fig4.update_layout(
                showlegend=False, font_family="DM Sans",
                margin=dict(l=0,r=0,t=10,b=40), height=340,
                yaxis=dict(title="Avg Hours/Week"),
            )
            st.plotly_chart(fig4, use_container_width=True)

        if "Gender" in df_clean.columns and "Sport" in df_clean.columns:
            st.markdown('<div class="section-header">Sport Participation by Gender</div>', unsafe_allow_html=True)
            cross = df_clean.groupby(["Sport","Gender"]).size().reset_index(name="Count")
            fig5 = px.bar(
                cross, x="Sport", y="Count", color="Gender",
                color_discrete_sequence=["#6366F1","#F59E0B","#10B981"],
                template="plotly_white",
                barmode="group",
                labels={"Count": "Participants", "Sport": ""},
            )
            fig5.update_layout(
                font_family="DM Sans",
                margin=dict(l=0,r=0,t=10,b=40), height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig5.update_traces(marker_line_width=0)
            st.plotly_chart(fig5, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Cleaned dataset preview</div>', unsafe_allow_html=True)
    display_cols = [c for c in df_clean.columns if c in
                    ["Sport","Gender","Age","Faculty","SportsHrs_num","Sleep_num",
                     "Stress_num","Height_num","ActivityLevel","AcademicWorkload"]]
    if not display_cols:
        display_cols = df_clean.columns.tolist()

    st.dataframe(df_clean[display_cols], use_container_width=True, height=420)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv_bytes, "sports_cleaned.csv", "text/csv")
    with col_dl2:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df_clean.to_excel(writer, index=False, sheet_name="Cleaned Data")
            log_df = pd.DataFrame(log, columns=["Step","Action","Detail","Rows After"])
            log_df.to_excel(writer, index=False, sheet_name="Cleaning Log")
        st.download_button("⬇ Download Excel", excel_buf.getvalue(),
                           "sports_cleaned.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if step5:
        st.markdown('<div class="section-header">Numeric summary statistics</div>', unsafe_allow_html=True)
        available_num = [c for c in NUM_COLS if c in df_clean.columns]
        if available_num:
            st.dataframe(
                df_clean[available_num].describe().round(2).rename(columns={
                    "SportsHrs_num": "SportsHrs",
                    "Sleep_num":     "Sleep",
                    "Stress_num":    "Stress",
                    "Height_num":    "Height (cm)",
                }),
                use_container_width=True,
            )


