import os

import anthropic
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Audit Risk Intelligence Dashboard",
    page_icon="📈",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CHART_PALETTE = [
    "#10B981", "#3B82F6", "#F59E0B", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#F97316",
]
TIER_CLR = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
AGING_CLR = {
    "1–30 days":   "#FEF9C3",
    "31–60 days":  "#FDE047",
    "61–90 days":  "#F97316",
    "91–180 days": "#DC2626",
    "180+ days":   "#7F1D1D",
}
REQUIRED_COLUMNS = [
    "Division", "Bureau", "Unit", "Program Name", "Audit_Title",
    "Finding_ID", "Finding_Theme", "Root_Cause", "Risk_Level", "Fiscal_Year",
    "Quarter", "Recommendation_Count", "CAP Days Overdue", "CAP Status",
    "Repeat_Finding", "Control_Type", "Notes",
]

# ── Session state ──────────────────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw    = None
    st.session_state.file_name = ""
if "ai_running" not in st.session_state:
    st.session_state.ai_running = False

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background-color: #F1F5F9; }
.main .block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.25rem !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > hr { display: none; }
[data-testid="stSidebar"] label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: #94A3B8 !important; text-transform: uppercase !important; letter-spacing: 0.6px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #ECFDF5 !important; color: #065F46 !important; border: none !important;
    border-radius: 6px !important; font-size: 0.7rem !important; font-weight: 500 !important;
}
[data-testid="stFileUploader"] > div {
    background: #F8FAFC !important; border: 1.5px dashed #CBD5E1 !important;
    border-radius: 12px !important; transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"] > div:hover { border-color: #10B981 !important; background: #F0FDF4 !important; }
[data-testid="stFileUploader"] small { color: #94A3B8 !important; font-size: 0.72rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important; border-radius: 14px !important; padding: 6px 8px !important;
    gap: 4px !important; border-bottom: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important; margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 10px !important; color: #64748B !important;
    font-weight: 500 !important; font-size: 0.84rem !important; padding: 8px 18px !important;
    border: none !important; transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { background: #F8FAFC !important; color: #1E293B !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: #FFFFFF !important; font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(16,185,129,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.875rem !important; padding: 0.6rem 2rem !important;
    box-shadow: 0 2px 10px rgba(16,185,129,0.35) !important; transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(16,185,129,0.40) !important;
}

/* ── Dataframe / misc ── */
[data-testid="stDataFrame"] > div {
    border-radius: 12px !important; border: 1px solid #E2E8F0 !important; overflow: hidden !important;
}
[data-testid="stAlert"] { border-radius: 12px !important; }
hr { border-color: #E2E8F0 !important; margin: 1rem 0 !important; }

/* ── Sidebar brand ── */
.sid-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.sid-icon {
    width: 40px; height: 40px; background: linear-gradient(135deg, #10B981, #059669);
    border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 19px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(16,185,129,0.28);
}
.sid-name  { font-size: 0.9rem; font-weight: 700; color: #0F172A; letter-spacing: -0.2px; line-height: 1.2; }
.sid-sub   { font-size: 0.68rem; color: #0F172A; font-weight: 500; }
.sid-label { font-size: 0.67rem; font-weight: 700; color: #94A3B8;
             text-transform: uppercase; letter-spacing: 0.7px; margin: 20px 0 8px 0; }
.sid-file-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
                 padding: 12px 14px; margin-bottom: 4px; }
.sid-file-name { font-size: 0.78rem; font-weight: 600; color: #1E293B; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
.sid-file-meta { font-size: 0.7rem; color: #64748B; }
.sid-file-dot  { color: #10B981; font-weight: 700; margin-right: 6px; }

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF; border-radius: 16px; padding: 18px 16px 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    border: 1px solid #F1F5F9; transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.10); }
.kpi-icon-wrap {
    width: 42px; height: 42px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; margin-bottom: 12px;
}
.ic-green  { background: #ECFDF5; } .ic-red    { background: #FEF2F2; }
.ic-orange { background: #FFF7ED; } .ic-amber  { background: #FFFBEB; }
.ic-purple { background: #F5F3FF; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #0F172A;
             line-height: 1; letter-spacing: -0.5px; margin-bottom: 4px; }
.kpi-label { font-size: 0.74rem; font-weight: 600; color: #64748B; margin-bottom: 9px; line-height: 1.3; }
.kpi-badge { display: inline-block; font-size: 0.66rem; font-weight: 600;
             padding: 2px 8px; border-radius: 20px; }
.b-green  { background: #DCFCE7; color: #166534; } .b-red    { background: #FEE2E2; color: #991B1B; }
.b-orange { background: #FEF3C7; color: #92400E; } .b-gray   { background: #F1F5F9; color: #475569; }
.b-purple { background: #EDE9FE; color: #5B21B6; }

/* ── Page header ── */
.rid-page-header { margin-bottom: 4px; }
.rid-page-header h1 {
    font-size: 1.45rem; font-weight: 800; color: #0F172A;
    margin: 0 0 2px 0; letter-spacing: -0.4px; line-height: 1.2;
}
.rid-page-header p { font-size: 0.81rem; color: #64748B; margin: 0 0 16px 0; }

/* ── Tab intro box ── */
.tab-intro {
    background: #FFFFFF; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 20px; font-size: 0.82rem; color: #475569; line-height: 1.6;
    border: 1px solid #E2E8F0;
}
.tab-intro strong { color: #0F172A; }

/* ── Priority cards ── */
.priority-header {
    font-size: 0.7rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.7px; margin: 0 0 10px 0;
}
.priority-card {
    background: #FFFFFF; border-radius: 0 14px 14px 0;
    border-left: 5px solid #EF4444;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    height: 100%;
}
.priority-rank-badge {
    display: inline-block; font-size: 0.63rem; font-weight: 700;
    color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.4px;
    padding: 3px 8px; border-radius: 5px; margin-bottom: 9px;
}
.priority-bureau {
    font-size: 0.92rem; font-weight: 700; color: #0F172A;
    line-height: 1.25; margin-bottom: 2px;
}
.priority-division { font-size: 0.7rem; color: #94A3B8; margin-bottom: 10px; }
.priority-score-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.76rem; color: #475569; margin-bottom: 10px;
}
.priority-score-num { font-size: 1.3rem; font-weight: 800; color: #0F172A; }
.priority-divider { width: 100%; height: 1px; background: #F1F5F9; margin: 8px 0; }
.priority-drivers { list-style: none; padding: 0; margin: 0; }
.priority-drivers li {
    font-size: 0.74rem; color: #475569; padding: 2px 0;
    display: flex; align-items: flex-start; gap: 6px;
}
.priority-drivers li::before { content: "→"; color: #10B981; font-weight: 700; flex-shrink: 0; }

/* ── Section header ── */
.rid-section-title { font-size: 0.94rem; font-weight: 700; color: #0F172A; margin: 0 0 2px 0; }
.rid-section-sub   { font-size: 0.76rem; color: #94A3B8; margin: 0 0 14px 0; }

/* ── Insight banner ── */
.insight-banner {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1px solid #A7F3D0; border-left: 4px solid #10B981;
    border-radius: 0 12px 12px 0; padding: 14px 20px; margin: 12px 0 20px 0;
    font-size: 0.86rem; color: #065F46; line-height: 1.65;
}
.insight-banner strong { color: #064E3B; }

/* ── Recommended action cards ── */
.action-card {
    background: #FFFFFF; border-radius: 12px; padding: 16px 16px 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    border: 1px solid #F1F5F9; height: 100%;
}
.action-type-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 3px 9px; border-radius: 20px; margin-bottom: 10px;
}
.action-bureau { font-size: 0.88rem; font-weight: 700; color: #0F172A; margin-bottom: 2px; }
.action-division { font-size: 0.68rem; color: #94A3B8; margin-bottom: 8px; }
.action-justification {
    font-size: 0.75rem; color: #475569; line-height: 1.55;
    border-top: 1px solid #F1F5F9; padding-top: 8px; margin-top: 4px;
}

/* ── Priority card enhancements ── */
.priority-audit-type {
    display: inline-block; font-size: 0.63rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 2px 8px; border-radius: 20px; margin-bottom: 8px;
}
.priority-why {
    font-size: 0.73rem; color: #475569; line-height: 1.5;
    font-style: italic; margin-top: 6px;
}

/* ── Programs Driving Risk table ── */
.prog-table-wrap {
    background: #FFFFFF; border-radius: 14px; overflow: hidden;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 14px rgba(0,0,0,0.05);
    margin-top: 4px;
}
.prog-table-head {
    display: grid;
    grid-template-columns: 28px 1fr 140px 80px 80px 80px 100px;
    gap: 0; padding: 9px 18px;
    background: #F8FAFC; border-bottom: 1px solid #E2E8F0;
    font-size: 0.65rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.6px;
}
.prog-table-row {
    display: grid;
    grid-template-columns: 28px 1fr 140px 80px 80px 80px 100px;
    gap: 0; padding: 11px 18px; align-items: center;
    border-bottom: 1px solid #F1F5F9; transition: background 0.12s;
}
.prog-table-row:last-child { border-bottom: none; }
.prog-table-row:hover { background: #F8FAFC; }
.prog-rank { font-size: 0.72rem; font-weight: 700; color: #CBD5E1; }
.prog-name { font-size: 0.83rem; font-weight: 600; color: #0F172A; line-height: 1.3; }
.prog-bureau { font-size: 0.69rem; color: #94A3B8; margin-top: 1px; }
.prog-score-cell { display: flex; align-items: center; gap: 7px; }
.prog-score-bar-wrap {
    flex: 1; height: 5px; background: #F1F5F9; border-radius: 3px; overflow: hidden;
}
.prog-score-bar { height: 100%; border-radius: 3px; }
.prog-score-num { font-size: 0.78rem; font-weight: 700; color: #0F172A; min-width: 28px; text-align: right; }
.prog-cell { font-size: 0.78rem; color: #475569; }
.prog-badge {
    display: inline-block; font-size: 0.63rem; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
}

/* ── Suppress top-right running indicator / stop button ── */
[data-testid="stStatusWidget"] { display: none !important; }
header [data-testid="stToolbar"] { display: none !important; }

/* ── AI loading card ── */
.ai-loading-card {
    background: #FFFFFF; border: 1px solid #A7F3D0; border-left: 4px solid #10B981;
    border-radius: 0 12px 12px 0; padding: 20px 24px; margin-top: 16px;
    display: flex; align-items: flex-start; gap: 16px;
}
.ai-loading-spinner {
    width: 20px; height: 20px; border: 2.5px solid #D1FAE5;
    border-top-color: #10B981; border-radius: 50%;
    animation: ai-spin 0.8s linear infinite; flex-shrink: 0; margin-top: 2px;
}
@keyframes ai-spin { to { transform: rotate(360deg); } }
.ai-loading-text-primary {
    font-size: 0.9rem; font-weight: 600; color: #064E3B; margin-bottom: 4px;
}
.ai-loading-text-secondary {
    font-size: 0.78rem; color: #065F46; line-height: 1.5;
}

/* ── AI box ── */
.rid-ai-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #10B981;
    border-radius: 0 12px 12px 0; padding: 24px 28px; margin-top: 20px;
    font-size: 0.88rem; line-height: 1.75; color: #1E293B;
}

/* ── Landing ── */
.landing-wrap {
    display: flex; flex-direction: column; align-items: center;
    text-align: center; padding: 52px 16px 20px 16px;
}
.landing-icon {
    width: 56px; height: 56px; background: linear-gradient(135deg, #10B981, #059669);
    border-radius: 16px; display: flex; align-items: center; justify-content: center;
    margin: 0 auto 14px auto; box-shadow: 0 6px 20px rgba(16,185,129,0.28);
}
.landing-title { font-size: 1.55rem; font-weight: 800; color: #0F172A;
                 letter-spacing: -0.4px; margin: 0 0 10px 0; line-height: 1.2;
                 white-space: nowrap; }
.landing-purpose { font-size: 0.9rem; font-weight: 400; color: #64748B;
                   margin: 0; letter-spacing: 0; line-height: 1.5; }
.landing-divider { height: 32px; }
.landing-upload-label { font-size: 1.05rem; font-weight: 700; color: #0F172A;
                        letter-spacing: -0.1px; margin: 0 0 14px 0; text-align: center; width: 100%; }
.landing-upload-wrap { width: 100%; }
.landing-upload-wrap [data-testid="stFileUploader"] > div {
    background: #FFFFFF !important;
    border: 2px dashed #10B981 !important;
    border-radius: 14px !important;
    padding: 28px 20px !important;
    box-shadow: 0 4px 24px rgba(16,185,129,0.10) !important;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
}
.landing-upload-wrap [data-testid="stFileUploader"] > div:hover {
    background: #F0FDF4 !important;
    box-shadow: 0 6px 32px rgba(16,185,129,0.18) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core data functions  (logic unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_data(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            return None, "Unsupported format. Upload a .xlsx or .csv file."
    except Exception as exc:
        return None, f"Error reading file: {exc}"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"

    df["Fiscal_Year"] = df["Fiscal_Year"].astype(str)
    df["CAP Days Overdue"] = pd.to_numeric(df["CAP Days Overdue"], errors="coerce").fillna(0).astype(int)
    df["Recommendation_Count"] = (
        pd.to_numeric(df["Recommendation_Count"], errors="coerce").fillna(0).astype(int)
    )
    return df, None


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    bureaus      = df["Bureau"].unique()
    current_year = df["Fiscal_Year"].astype(int).max()
    max_findings = df.groupby("Bureau").size().max()
    SEV = {"High": 3, "Medium": 2, "Low": 1}

    records = []
    for bureau in bureaus:
        bdf      = df[df["Bureau"] == bureau]
        division = bdf["Division"].iloc[0]
        open_bdf = bdf[bdf["CAP Status"] == "Open"]

        severity_norm = (bdf["Risk_Level"].map(SEV).mean() - 1) / 2
        volume_norm   = len(bdf) / max_findings
        age_norm      = (min(open_bdf["CAP Days Overdue"].mean(), 365) / 365
                         if len(open_bdf) > 0 else 0.0)
        repeat_norm   = (bdf["Repeat_Finding"] == "Yes").mean()
        last_year     = bdf["Fiscal_Year"].astype(int).max()
        recency_norm  = min(current_year - last_year, 3) / 3

        score = (severity_norm * 0.35 + volume_norm * 0.25 + age_norm * 0.20
                 + repeat_norm * 0.12 + recency_norm * 0.08) * 100
        tier  = "High" if score >= 60 else ("Medium" if score >= 35 else "Low")

        records.append({
            "Division":           division,
            "Bureau":             bureau,
            "Risk Score":         round(score, 1),
            "Risk Tier":          tier,
            "Open High Findings": int(((bdf["CAP Status"]=="Open") & (bdf["Risk_Level"]=="High")).sum()),
            "Repeat Finding %":   round(repeat_norm * 100, 1),
            "Last Audited":       last_year,
            "_open_avg_days":     open_bdf["CAP Days Overdue"].mean() if len(open_bdf) > 0 else 0,
        })

    result = (pd.DataFrame(records)
              .sort_values("Risk Score", ascending=False)
              .reset_index(drop=True))
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


def build_ai_prompt(df: pd.DataFrame, ranked: pd.DataFrame) -> str:
    n         = len(df)
    open_ct   = int((df["CAP Status"] == "Open").sum())
    repeat_ct = int((df["Repeat_Finding"] == "Yes").sum())
    repeat_pct = repeat_ct / n * 100 if n > 0 else 0

    overdue_mask = (df["CAP Status"] == "Open") & (df["CAP Days Overdue"] > 0)
    overdue_ct   = int(overdue_mask.sum())
    avg_overdue  = df.loc[overdue_mask, "CAP Days Overdue"].mean() if overdue_ct > 0 else 0

    yearly = (df.groupby("Fiscal_Year").agg(
        total  =("Finding_ID",    "count"),
        high   =("Risk_Level",    lambda x: (x=="High").sum()),
        repeat =("Repeat_Finding",lambda x: (x=="Yes").sum()),
        open   =("CAP Status",        lambda x: (x=="Open").sum()),
    ).sort_index())
    yearly_txt = "\n".join(
        f"  FY{yr}: {r['total']} total | {r['high']} high-risk | "
        f"{r['repeat']} repeat | {r['open']} open"
        for yr, r in yearly.iterrows()
    )

    top5_risk = ranked.head(5)
    bureau_lines = []
    for _, row in top5_risk.iterrows():
        bdf    = df[df["Bureau"] == row["Bureau"]]
        themes = bdf["Finding_Theme"].value_counts().head(3).to_dict()
        roots  = bdf["Root_Cause"].value_counts().head(3).to_dict()
        bureau_lines.append(
            f"  • {row['Bureau']} ({row['Division']}) — Score {row['Risk Score']} "
            f"({row['Risk Tier']}), Open High: {row['Open High Findings']}, "
            f"Repeat: {row['Repeat Finding %']}%, Avg Days Overdue: {row['_open_avg_days']:.0f}\n"
            f"    Themes: {', '.join(f'{k}({v})' for k,v in themes.items())}\n"
            f"    Root causes: {', '.join(f'{k}({v})' for k,v in roots.items())}"
        )

    top5_repeat  = ranked.nlargest(5, "Repeat Finding %")
    repeat_lines = [f"  • {r['Bureau']}: {r['Repeat Finding %']:.1f}% repeat rate"
                    for _, r in top5_repeat.iterrows()]

    overdue_bur = (df[overdue_mask].groupby("Bureau")["CAP Days Overdue"]
                   .agg(count="count", avg="mean")
                   .sort_values("count", ascending=False).head(5))
    overdue_lines = [f"  • {b}: {r['count']} items overdue (avg {r['avg']:.0f} days)"
                     for b, r in overdue_bur.iterrows()]

    theme_lines = [f"  • {t}: {c} open findings"
                   for t, c in (df[df["CAP Status"]=="Open"]["Finding_Theme"]
                                .value_counts().head(5).items())]

    return f"""You are a senior internal audit risk analyst at the NYC Department of Health and Mental Hygiene (DOHMH) preparing a briefing for the Deputy Commissioner and Chief Audit Executive.

PORTFOLIO OVERVIEW:
- Total findings: {n:,} | Open: {open_ct} ({open_ct/n*100:.1f}%) | Repeat: {repeat_ct} ({repeat_pct:.1f}%)
- Overdue open items: {overdue_ct} (avg {avg_overdue:.0f} days past due)
- Bureaus assessed: {len(ranked)} across {df['Division'].nunique()} divisions

YEAR-OVER-YEAR TREND:
{yearly_txt}

TOP 5 HIGHEST-RISK BUREAUS:
{chr(10).join(bureau_lines)}

TOP THEMES IN OPEN FINDINGS:
{chr(10).join(theme_lines)}

TOP BUREAUS BY REPEAT FINDING RATE:
{chr(10).join(repeat_lines)}

TOP BUREAUS BY OVERDUE CORRECTIVE ACTIONS:
{chr(10).join(overdue_lines)}

Write a professional internal audit risk narrative using exactly these section headings:

## Executive Summary
2–3 sentences on overall risk posture.

## Top Priority Areas for Audit Attention
The 3–5 most critical bureaus or divisions that require immediate audit focus, and why.

## Key Risk Themes
The dominant types of control failures appearing across the organization, with patterns noted.

## Repeat Issue Patterns
Which areas have persistent findings that are not being remediated, and what this signals about control culture.

## Aging Corrective Actions
Which bureaus have the most concerning CAP backlogs, and the operational risk this creates.

## Suggested Audit Planning Takeaways
4–5 specific, actionable recommendations for the next fiscal year audit plan.

Be specific, cite bureau and division names, and write in a tone appropriate for a Deputy Commissioner or Chief Audit Executive. Do not pad with generic statements."""


# ── Chart helpers ──────────────────────────────────────────────────────────────

def chart_base(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", color="#111111", size=13),
        margin=dict(l=10, r=10, t=36, b=10),
    )
    base.update(kwargs)
    return base

AXIS_DEF   = dict(showgrid=False, linecolor="#E2E8F0", tickfont=dict(size=12, color="#111111"))
YAXIS_DEF  = dict(gridcolor="#F1F5F9", gridwidth=1, linecolor="#E2E8F0", tickfont=dict(size=12, color="#111111"))
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12, color="#111111"))


# ── UI component helpers ───────────────────────────────────────────────────────

def kpi_card(icon, icon_cls, value, label, badge="", badge_cls="b-gray"):
    bdg = f'<span class="kpi-badge {badge_cls}">{badge}</span>' if badge else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-icon-wrap {icon_cls}">{icon}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>{bdg}</div>')


def render_kpis(df: pd.DataFrame, ranked: pd.DataFrame):
    n         = len(df)
    open_ct   = int((df["CAP Status"] == "Open").sum())
    closed_ct = n - open_ct
    open_high = int(((df["CAP Status"] == "Open") & (df["Risk_Level"] == "High")).sum())
    repeat_ct = int((df["Repeat_Finding"] == "Yes").sum())
    repeat_pct = repeat_ct / n * 100 if n > 0 else 0.0
    ov_mask    = (df["CAP Status"] == "Open") & (df["CAP Days Overdue"] > 0)
    overdue_ct = int(ov_mask.sum())
    avg_od     = df.loc[ov_mask, "CAP Days Overdue"].mean() if overdue_ct > 0 else 0.0
    total_b    = len(ranked) if not ranked.empty else df["Bureau"].nunique()
    elevated   = int((ranked["Risk Tier"].isin(["High","Medium"])).sum()) if not ranked.empty else 0

    cards = [
        kpi_card("📋","ic-green",  f"{n:,}",           "Total Findings",
                 f"{open_ct} open · {closed_ct} closed", "b-gray"),
        kpi_card("🚨","ic-red",    str(open_high),      "Critical Issues Requiring Action",
                 "Immediate escalation needed" if open_high > 0 else "None open",
                 "b-red" if open_high > 5 else "b-orange" if open_high > 0 else "b-green"),
        kpi_card("🔁","ic-orange", f"{repeat_pct:.1f}%","Recurring Issues (Systemic Risk)",
                 f"{repeat_ct} of {n} not remediated",
                 "b-red" if repeat_pct >= 30 else "b-orange" if repeat_pct >= 15 else "b-green"),
        kpi_card("⏱️","ic-amber",  str(overdue_ct),     "Overdue CAPs",
                 f"Avg {avg_od:.0f} days past due" if overdue_ct > 0 else "All on track",
                 "b-red" if overdue_ct > 20 else "b-orange" if overdue_ct > 10 else "b-green"),
        kpi_card("🏢","ic-purple", f"{elevated} / {total_b}", "Bureaus Needing Attention",
                 "High or Medium risk",
                 "b-red"    if total_b > 0 and elevated / total_b > 0.6
                 else "b-orange" if total_b > 0 and elevated / total_b > 0.35 else "b-green"),
    ]
    cols = st.columns(5)
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


def tab_intro(text: str):
    st.markdown(f'<div class="tab-intro">{text}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    sub = f'<p class="rid-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="rid-section-title">{title}</div>{sub}', unsafe_allow_html=True)


def spacer():
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def get_key_drivers(row: pd.Series, df: pd.DataFrame) -> list:
    """Auto-generate 2–3 plain-language risk drivers for a bureau."""
    drivers = []
    if row["Open High Findings"] > 0:
        n = row["Open High Findings"]
        drivers.append(f"{n} open critical finding{'s' if n > 1 else ''}")
    if row["Repeat Finding %"] >= 20:
        drivers.append(f"{row['Repeat Finding %']:.0f}% of issues have recurred before")
    if row["_open_avg_days"] >= 60:
        drivers.append(f"Corrective actions avg {row['_open_avg_days']:.0f} days overdue")
    top_theme = (df[df["Bureau"] == row["Bureau"]]["Finding_Theme"]
                 .value_counts().index)
    if len(top_theme) > 0 and len(drivers) < 3:
        drivers.append(f"Top issue area: {top_theme[0]}")
    if len(drivers) == 0:
        drivers.append(f"Risk score {row['Risk Score']:.0f} — elevated overall exposure")
    return drivers[:3]


def suggest_audit_type(row: pd.Series) -> tuple:
    """Return (label, hex_color) for the recommended audit type."""
    if row["Repeat Finding %"] >= 25:
        return "Follow-up Audit", "#8B5CF6"
    if row["Open High Findings"] >= 3 or row["Risk Score"] >= 70:
        return "Full Audit", "#EF4444"
    if row["_open_avg_days"] >= 90:
        return "Advisory Review", "#F59E0B"
    return "Risk Assessment", "#3B82F6"


def get_why_matters(row: pd.Series, df: pd.DataFrame) -> str:
    """Return a single-sentence explanation of the bureau's key risk signal."""
    if row["Repeat Finding %"] >= 25:
        return f"{row['Repeat Finding %']:.0f}% recurrence rate signals a persistent control failure, not a one-time issue."
    if row["Open High Findings"] >= 3:
        return f"{row['Open High Findings']} unresolved critical findings represent direct operational and compliance exposure."
    if row["_open_avg_days"] >= 90:
        return f"Corrective actions averaging {row['_open_avg_days']:.0f} days overdue suggests prior recommendations are not being actioned."
    top_theme = df[df["Bureau"] == row["Bureau"]]["Finding_Theme"].value_counts()
    if len(top_theme) > 0:
        return f"Concentrated {top_theme.index[0]} findings with a risk score of {row['Risk Score']:.0f} warrant structured audit attention."
    return f"Elevated risk score of {row['Risk Score']:.0f} across multiple dimensions requires audit follow-through."


def generate_recommended_actions(ranked: pd.DataFrame, df: pd.DataFrame) -> list:
    """Generate up to 5 recommended audit actions ranked by risk signal strength."""
    actions = []
    for _, row in ranked.iterrows():
        parts = []
        if row["Repeat Finding %"] >= 20:
            parts.append(f"{row['Repeat Finding %']:.0f}% repeat rate")
        if row["Open High Findings"] > 0:
            parts.append(f"{row['Open High Findings']} open critical finding{'s' if row['Open High Findings'] > 1 else ''}")
        if row["_open_avg_days"] >= 60:
            parts.append(f"avg {row['_open_avg_days']:.0f} days overdue on CAPs")
        if not parts:
            parts.append(f"risk score of {row['Risk Score']:.0f}")
        audit_type, _ = suggest_audit_type(row)
        actions.append({
            "bureau":       row["Bureau"],
            "division":     row["Division"],
            "audit_type":   audit_type,
            "score":        row["Risk Score"],
            "justification": ("; ".join(parts[:2]) + ".").capitalize(),
        })
        if len(actions) >= 5:
            break
    return actions


def generate_summary_insight(df: pd.DataFrame, ranked: pd.DataFrame) -> str:
    """Generate a 1–2 sentence risk insight from the loaded data."""
    open_ct    = int((df["CAP Status"] == "Open").sum())
    repeat_pct = (df["Repeat_Finding"] == "Yes").mean() * 100
    ov_mask    = (df["CAP Status"] == "Open") & (df["CAP Days Overdue"] > 0)
    overdue_ct = int(ov_mask.sum())
    high_ct    = int((ranked["Risk Tier"] == "High").sum()) if not ranked.empty else 0
    top_bureau = ranked.iloc[0]["Bureau"] if not ranked.empty else "N/A"
    top_div    = ranked.iloc[0]["Division"] if not ranked.empty else ""
    n_div      = df["Division"].nunique()

    s1 = (f"<strong>{high_ct} bureau{'s are' if high_ct != 1 else ' is'} currently rated High Risk</strong> "
          f"with {open_ct:,} open findings across {n_div} divisions.")

    if repeat_pct >= 20:
        s2 = (f"A <strong>{repeat_pct:.0f}% repeat finding rate</strong> signals systemic control failures "
              f"that require structured follow-up, not one-off remediation.")
    elif overdue_ct >= 10:
        s2 = (f"<strong>{overdue_ct} corrective action plans are overdue</strong>, indicating prior audit "
              f"recommendations are not being implemented on schedule.")
    else:
        s2 = (f"<strong>{top_bureau}</strong> ({top_div}) carries the highest risk score "
              f"and should be the immediate focus for audit planning.")

    return f"{s1} {s2}"


def render_recommended_actions(actions: list):
    """Render the recommended audit actions section."""
    st.markdown(
        '<div class="rid-section-title">Recommended Audit Actions — Next 90 Days</div>'
        '<div class="rid-section-sub" style="margin-bottom:14px">'
        'Automatically prioritized by risk score, recurrence rate, and overdue corrective actions</div>',
        unsafe_allow_html=True)

    type_colors = {
        "Full Audit":      "#EF4444",
        "Follow-up Audit": "#8B5CF6",
        "Advisory Review": "#F59E0B",
        "Risk Assessment": "#3B82F6",
    }
    cols = st.columns(len(actions))
    for col, action in zip(cols, actions):
        color = type_colors.get(action["audit_type"], "#64748B")
        col.markdown(f"""
        <div class="action-card" style="border-top:3px solid {color}">
            <span class="action-type-badge" style="background:{color}18;color:{color}">
                {action['audit_type']}
            </span>
            <div class="action-bureau">{action['bureau']}</div>
            <div class="action-division">{action['division']}</div>
            <div class="action-justification">{action['justification']}</div>
        </div>""", unsafe_allow_html=True)


def render_coverage_chart(ranked: pd.DataFrame):
    """Scatter: Risk Score vs Last Audit Year — highlights under-audited high-risk bureaus."""
    section("Audit Coverage vs. Risk Exposure",
            "Bubble size = open critical issues. Bureaus in the top-left are high risk but not recently audited.")

    current_year = int(ranked["Last Audited"].max())

    fig = go.Figure()
    for tier in ["High", "Medium", "Low"]:
        sub = ranked[ranked["Risk Tier"] == tier].copy()
        if sub.empty:
            continue
        bubble_sizes = (sub["Open High Findings"].clip(lower=0) * 6 + 10).tolist()
        fig.add_trace(go.Scatter(
            x=sub["Last Audited"],
            y=sub["Risk Score"],
            mode="markers+text",
            name=f"{tier} Risk",
            text=sub["Bureau"],
            textposition="top center",
            textfont=dict(size=11, color="#111111"),
            marker=dict(
                color=TIER_CLR[tier],
                size=bubble_sizes,
                opacity=0.80,
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Last Audited: %{x}<br>"
                "Risk Score: %{y:.1f}<extra></extra>"
            ),
        ))

    # Shade the danger zone: high risk + stale audit
    fig.add_vrect(
        x0=ranked["Last Audited"].min() - 0.5,
        x1=current_year - 1.5,
        fillcolor="#FEE2E2", opacity=0.25, line_width=0,
        annotation_text="Not recently audited", annotation_position="top left",
        annotation_font=dict(size=10, color="#DC2626"),
    )

    fig.update_layout(
        **chart_base(height=360, margin=dict(l=10, r=10, t=40, b=10)),
        xaxis=dict(**AXIS_DEF, title=dict(text="Last Audit Year", font=dict(size=14, color="#111111")), dtick=1),
        yaxis=dict(**YAXIS_DEF, title=dict(text="Risk Score (0–100)", font=dict(size=14, color="#111111")), range=[0, 108]),
        legend=LEGEND_TOP,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_priority_cards(ranked: pd.DataFrame, df: pd.DataFrame):
    """Render the top 3 priority bureau cards."""
    st.markdown('<div class="priority-header">Recommended Audit Priorities — Act on These First</div>',
                unsafe_allow_html=True)

    rank_meta = [
        (1, "#EF4444", "1st Priority"),
        (2, "#F97316", "2nd Priority"),
        (3, "#F59E0B", "3rd Priority"),
    ]
    cols = st.columns(3)
    for col, (rank, color, label) in zip(cols, rank_meta):
        if rank - 1 >= len(ranked):
            continue
        row          = ranked.iloc[rank - 1]
        drivers      = get_key_drivers(row, df)
        tier_cls     = "b-red" if row["Risk Tier"] == "High" else "b-orange"
        audit_type, at_color = suggest_audit_type(row)
        why          = get_why_matters(row, df)
        drivers_html = "".join(f"<li>{d}</li>" for d in drivers)
        col.markdown(f"""
        <div class="priority-card" style="border-left-color:{color}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
                <span class="priority-rank-badge" style="background:{color};margin:0">{label}</span>
                <span class="priority-audit-type" style="background:{at_color}18;color:{at_color}">
                    {audit_type}
                </span>
            </div>
            <div class="priority-bureau">{row['Bureau']}</div>
            <div class="priority-division">{row['Division']}</div>
            <div class="priority-score-row">
                <span class="priority-score-num">{row['Risk Score']:.0f}</span>
                <span style="font-size:0.72rem;color:#94A3B8">/ 100 risk score</span>
                <span class="kpi-badge {tier_cls}" style="margin-left:6px">{row['Risk Tier']}</span>
            </div>
            <div class="priority-divider"></div>
            <ul class="priority-drivers">{drivers_html}</ul>
            <div class="priority-why">{why}</div>
        </div>""", unsafe_allow_html=True)


def compute_program_scores(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Score and rank programs (Program Name) using the same factor logic as bureaus."""
    if df.empty or "Program Name" not in df.columns:
        return pd.DataFrame()

    programs     = df["Program Name"].dropna().unique()
    max_findings = max(df.groupby("Program Name").size().max(), 1)
    SEV          = {"High": 3, "Medium": 2, "Low": 1}

    records = []
    for prog in programs:
        pdf      = df[df["Program Name"] == prog]
        open_pdf = pdf[pdf["CAP Status"] == "Open"]

        severity_norm = (pdf["Risk_Level"].map(SEV).mean() - 1) / 2
        volume_norm   = len(pdf) / max_findings
        age_norm      = (min(open_pdf["CAP Days Overdue"].mean(), 365) / 365
                         if len(open_pdf) > 0 else 0.0)
        repeat_norm   = (pdf["Repeat_Finding"] == "Yes").mean()

        # Programs don't have a recency dimension — redistribute weight evenly
        score = (severity_norm * 0.38 + volume_norm * 0.27 +
                 age_norm * 0.22 + repeat_norm * 0.13) * 100

        open_high = int(((pdf["CAP Status"] == "Open") & (pdf["Risk_Level"] == "High")).sum())
        avg_days  = round(open_pdf["CAP Days Overdue"].mean(), 0) if len(open_pdf) > 0 else 0

        records.append({
            "Program Name":      prog,
            "Bureau":            pdf["Bureau"].iloc[0],
            "Division":          pdf["Division"].iloc[0],
            "Risk Score":        round(score, 1),
            "Open Critical":     open_high,
            "Repeat Rate":       round(repeat_norm * 100, 1),
            "Avg Days Overdue":  int(avg_days),
        })

    return (pd.DataFrame(records)
            .sort_values("Risk Score", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


def render_program_risk_table(prog_df: pd.DataFrame):
    """Render the Programs Driving Risk Exposure section."""
    section("Programs Driving Risk Exposure",
            "Top program areas contributing to risk within the current selection.")

    if prog_df.empty:
        st.info("No program data available for the current selection.")
        return

    tier_color = lambda s: "#EF4444" if s >= 60 else ("#F59E0B" if s >= 35 else "#10B981")
    badge_cls  = lambda s: "b-red"   if s >= 60 else ("b-orange" if s >= 35 else "b-green")

    header_html = """
    <div class="prog-table-wrap">
      <div class="prog-table-head">
        <div></div>
        <div>Program</div>
        <div>Division</div>
        <div>Risk Score</div>
        <div>Open Critical</div>
        <div>Repeat Rate</div>
        <div>Avg Days Overdue</div>
      </div>"""

    rows_html = ""
    for i, row in prog_df.iterrows():
        rank   = i + 1
        score  = row["Risk Score"]
        color  = tier_color(score)
        bcls   = badge_cls(score)
        bar_w  = int(score)
        od     = row["Avg Days Overdue"]
        od_cls = "b-red" if od >= 90 else ("b-orange" if od >= 30 else "b-gray")

        rows_html += f"""
      <div class="prog-table-row">
        <div class="prog-rank">#{rank}</div>
        <div>
          <div class="prog-name">{row['Program Name']}</div>
          <div class="prog-bureau">{row['Bureau']}</div>
        </div>
        <div style="font-size:0.75rem;color:#475569">{row['Division']}</div>
        <div class="prog-score-cell">
          <div class="prog-score-bar-wrap">
            <div class="prog-score-bar" style="width:{bar_w}%;background:{color}"></div>
          </div>
          <span class="prog-score-num" style="color:{color}">{score:.0f}</span>
        </div>
        <div class="prog-cell">
          <span class="prog-badge {bcls if row['Open Critical'] > 0 else 'b-gray'}">
            {row['Open Critical']}
          </span>
        </div>
        <div class="prog-cell">{row['Repeat Rate']:.0f}%</div>
        <div class="prog-cell">
          <span class="prog-badge {od_cls}">{od}d</span>
        </div>
      </div>"""

    st.markdown(header_html + rows_html + "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

df = None

with st.sidebar:
    st.markdown("""
    <div class="sid-brand">
        <div class="sid-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="20" height="20"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg></div>
        <div><div class="sid-name">NYC DOHMH</div>
        <div class="sid-sub">Audit Services</div></div>
    </div>""", unsafe_allow_html=True)

    df_raw = st.session_state.df_raw

    if df_raw is not None:
        st.markdown('<div class="sid-label">Filter Results</div>', unsafe_allow_html=True)
        all_yr  = sorted(df_raw["Fiscal_Year"].dropna().unique())
        sel_yr  = st.multiselect("Fiscal Year", all_yr,  default=[])

        all_div = sorted(df_raw["Division"].dropna().unique())
        sel_div = st.multiselect("Division",    all_div, default=[])

        # Bureau list scoped to selected divisions (or all if none chosen)
        div_scope   = sel_div if sel_div else all_div
        all_bureau  = sorted(df_raw[df_raw["Division"].isin(div_scope)]["Bureau"].dropna().unique())
        sel_bureau  = st.multiselect("Bureau",      all_bureau, default=[])

        df = df_raw.copy()
        if sel_yr:     df = df[df["Fiscal_Year"].isin(sel_yr)]
        if sel_div:    df = df[df["Division"].isin(sel_div)]
        if sel_bureau: df = df[df["Bureau"].isin(sel_bureau)]

        with st.expander("Replace data file"):
            new_file = st.file_uploader("New file", type=["xlsx","csv"],
                                        key="sidebar_upload", label_visibility="collapsed")
            if new_file:
                new_df, err = load_data(new_file)
                if err:
                    st.error(err)
                else:
                    st.session_state.df_raw    = new_df
                    st.session_state.file_name = new_file.name
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Landing
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.df_raw is None:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    </style>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div class="landing-wrap">
            <div class="landing-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"
                     width="28" height="28">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <polyline points="8 11 10.5 13.5 16 8"/>
                </svg>
            </div>
            <div class="landing-title">Audit Risk Intelligence Dashboard</div>
            <div class="landing-purpose">Identify where audit resources should be prioritized across the agency.</div>
            <div class="landing-divider"></div>
            <div class="landing-upload-label">Upload Audit Findings Data to Begin Analysis</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="landing-upload-wrap">', unsafe_allow_html=True)
        center_file = st.file_uploader("Upload", type=["xlsx","csv"],
                                       key="center_upload", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if center_file:
            with st.spinner("Loading…"):
                new_df, err = load_data(center_file)
            if err:
                st.error(err)
            else:
                st.session_state.df_raw    = new_df
                st.session_state.file_name = center_file.name
                st.rerun()
    st.stop()

if df is None:
    df = st.session_state.df_raw.copy()
if len(df) == 0:
    st.warning("No findings match the current filter selection. Adjust the sidebar filters.")
    st.stop()

# Compute once — reused by all tabs
ranked = compute_risk_scores(df)


# ══════════════════════════════════════════════════════════════════════════════
# Page header + KPIs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="rid-page-header">
    <h1>Audit Risk Intelligence Dashboard</h1>
</div>""", unsafe_allow_html=True)

render_kpis(df, ranked)
spacer()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_landscape, tab_repeat, tab_trends, tab_heat, tab_ai, tab_data = st.tabs([
    "  📊  Risk Overview  ",
    "  🗺️  Agency Risk Landscape  ",
    "  🔁  Repeat & Aging  ",
    "  📈  Trends  ",
    "  🔥  Issue Heat Map  ",
    "  🤖  AI Briefing  ",
    "  📋  Raw Data  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Risk Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Summary insight banner ─────────────────────────────────────────────────
    insight_text = generate_summary_insight(df, ranked)
    st.markdown(f'<div class="insight-banner">💡 {insight_text}</div>',
                unsafe_allow_html=True)

    # ── Top 3 priority cards ───────────────────────────────────────────────────
    spacer()
    st.markdown('<div class="priority-header">Top Priority Bureaus — Ranked by Risk Exposure</div>',
                unsafe_allow_html=True)
    render_priority_cards(ranked, df)

    # ── Programs driving risk ──────────────────────────────────────────────────
    spacer()
    prog_scores = compute_program_scores(df)
    render_program_risk_table(prog_scores)

    # ── Recommended audit actions ──────────────────────────────────────────────
    spacer()
    actions = generate_recommended_actions(ranked, df)
    render_recommended_actions(actions)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Agency Risk Landscape
# ══════════════════════════════════════════════════════════════════════════════

with tab_landscape:

    st.markdown("""
    <div class="rid-page-header" style="margin-bottom:16px">
        <h1 style="font-size:1.2rem;margin-bottom:4px">Agency Risk Landscape</h1>
        <p>A consolidated view of risk distribution, issue types, and audit coverage across the organization.</p>
    </div>""", unsafe_allow_html=True)

    # ── All bureaus ranked bar ─────────────────────────────────────────────────
    section("All Bureaus Ranked by Risk Score",
            "Score is 0–100. Red = high risk, amber = medium, green = lower risk.")

    top10 = ranked.head(10).sort_values("Risk Score", ascending=True)
    fig_top10 = go.Figure(go.Bar(
        x=top10["Risk Score"],
        y=top10["Bureau"],
        orientation="h",
        marker_color=[TIER_CLR[t] for t in top10["Risk Tier"]],
        marker_line_width=0,
        text=top10["Risk Score"].apply(lambda v: f"{v:.0f}"),
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter"),
        customdata=top10[["Division","Risk Tier","Open High Findings",
                           "Repeat Finding %","_open_avg_days"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Division: %{customdata[0]}<br>"
            "Score: %{x:.1f} — %{customdata[1]} Risk<br>"
            "Open Critical Issues: %{customdata[2]}<br>"
            "Recurrence Rate: %{customdata[3]:.1f}%<br>"
            "Avg Days Overdue: %{customdata[4]:.0f}<extra></extra>"
        ),
    ))
    fig_top10.update_layout(
        **chart_base(height=350, margin=dict(l=230, r=60, t=10, b=10)),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="#F1F5F9",
                   title=dict(text="Risk Score (0–100)", font=dict(size=14, color="#111111")),
                   tickfont=dict(size=12, color="#111111")),
        yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color="#111111")),
    )
    st.plotly_chart(fig_top10, use_container_width=True)

    spacer()

    # ── Issue types + Risk tier donut ──────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("Issue Types by Open Finding Volume",
                "Bar length = open findings. Color intensity = proportion that are critical.")

        theme_stats = (
            df[df["CAP Status"] == "Open"]
            .groupby("Finding_Theme")
            .agg(Open    =("Finding_ID", "count"),
                 High_Pct=("Risk_Level", lambda x: (x == "High").mean() * 100))
            .reset_index()
            .sort_values("Open", ascending=True)
        )
        fig_theme = go.Figure(go.Bar(
            x=theme_stats["Open"],
            y=theme_stats["Finding_Theme"],
            orientation="h",
            marker=dict(
                color=theme_stats["High_Pct"],
                colorscale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#EF4444"]],
                cmin=0, cmax=100,
                colorbar=dict(title=dict(text="% Critical", font=dict(size=12, color="#111111")),
                          thickness=12, len=0.7, tickfont=dict(size=11, color="#111111")),
            ),
            marker_line_width=0,
            text=theme_stats["Open"],
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{y}</b><br>Open: %{x}<br>% Critical: %{marker.color:.0f}%<extra></extra>",
        ))
        fig_theme.update_layout(
            **chart_base(height=310, margin=dict(l=170, r=90, t=10, b=10)),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9",
                       title=dict(text="Open Findings", font=dict(size=14, color="#111111")),
                       tickfont=dict(size=12, color="#111111")),
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color="#111111")),
        )
        st.plotly_chart(fig_theme, use_container_width=True)

    with col_r:
        section("Risk Tier Distribution",
                f"{len(ranked)} bureaus across High, Medium, and Low risk categories.")

        tier_order  = ["High", "Medium", "Low"]
        tier_counts = ranked["Risk Tier"].value_counts()
        tier_vals   = [tier_counts.get(t, 0) for t in tier_order]

        fig_donut = go.Figure(go.Pie(
            labels=["High Risk","Medium Risk","Low Risk"],
            values=tier_vals, hole=0.58,
            marker_colors=[TIER_CLR[t] for t in tier_order],
            textinfo="label+value",
            textfont=dict(size=13, color="#111111"),
            hovertemplate="<b>%{label}</b><br>%{value} bureaus<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            **chart_base(height=280, margin=dict(l=10, r=10, t=10, b=10)),
            showlegend=False,
            annotations=[dict(
                text=f"{len(ranked)}<br><span style='font-size:10px;color:#64748B'>bureaus</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#0F172A", family="Inter"),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Coverage vs Risk ───────────────────────────────────────────────────────
    spacer()
    render_coverage_chart(ranked)

    # ── Full ranking table ─────────────────────────────────────────────────────
    with st.expander("See full bureau ranking table"):
        display_cols = ["Rank","Division","Bureau","Risk Score","Risk Tier",
                        "Open High Findings","Repeat Finding %","Last Audited"]
        st.dataframe(
            ranked[display_cols],
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=100, format="%.1f"),
                "Open High Findings": st.column_config.NumberColumn("Critical Open Issues"),
                "Repeat Finding %":   st.column_config.NumberColumn(
                    "Recurrence Rate", format="%.1f%%"),
            },
            hide_index=True, use_container_width=True, height=420,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Repeat & Aging
# ══════════════════════════════════════════════════════════════════════════════

with tab_repeat:

    tab_intro(
        "<strong>Two of the most important signals for audit prioritization:</strong> "
        "recurring issues (findings that were raised before but never fully fixed) and "
        "aging corrective actions (commitments to remediate that are now overdue). "
        "High rates in either area indicate a culture where audit findings are not being acted on."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        section("Recurring Issues by Division",
                "What percentage of each division's findings have come up before. "
                "Higher = systemic problem, not a one-time lapse.")

        rep_div = (
            df.groupby("Division")
            .agg(Total  =("Finding_ID",     "count"),
                 Repeats =("Repeat_Finding", lambda x: (x=="Yes").sum()),
                 Rate    =("Repeat_Finding", lambda x: (x=="Yes").mean() * 100))
            .reset_index()
            .sort_values("Rate", ascending=True)
        )
        fig_rdiv = go.Figure(go.Bar(
            x=rep_div["Rate"],
            y=rep_div["Division"],
            orientation="h",
            marker=dict(color=rep_div["Rate"],
                        colorscale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#EF4444"]],
                        cmin=0, cmax=60),
            marker_line_width=0,
            text=rep_div.apply(lambda r: f"{int(r['Repeats'])} of {int(r['Total'])}", axis=1),
            textposition="outside",
            textfont=dict(size=11, color="#111111"),
            hovertemplate="<b>%{y}</b><br>Recurrence Rate: %{x:.1f}%<br>Count: %{text}<extra></extra>",
        ))
        x_max = max(rep_div["Rate"].max() * 1.35, 10)
        fig_rdiv.update_layout(
            **chart_base(height=420, margin=dict(l=240, r=100, t=10, b=30)),
            xaxis=dict(range=[0, x_max], showgrid=True, gridcolor="#F1F5F9",
                       title=dict(text="Recurrence Rate (%)", font=dict(size=14, color="#111111")),
                       tickfont=dict(size=12, color="#111111")),
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color="#111111")),
        )
        st.plotly_chart(fig_rdiv, use_container_width=True)

    with col_r:
        section("Bureaus With the Most Recurring Issues",
                "Count of repeat findings per bureau. Color shows the recurrence rate — "
                "red means most findings in that bureau have come back before.")

        rep_bur = (
            df.groupby("Bureau")
            .agg(Division=("Division",      "first"),
                 Total   =("Finding_ID",     "count"),
                 Repeats =("Repeat_Finding", lambda x: (x=="Yes").sum()),
                 Rate    =("Repeat_Finding", lambda x: (x=="Yes").mean() * 100))
            .reset_index()
            .nlargest(12, "Repeats")
            .sort_values("Repeats", ascending=True)
        )
        fig_rbur = go.Figure(go.Bar(
            x=rep_bur["Repeats"],
            y=rep_bur["Bureau"],
            orientation="h",
            marker=dict(color=rep_bur["Rate"],
                        colorscale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#EF4444"]],
                        cmin=0, cmax=60,
                        colorbar=dict(title=dict(text="Rate %", font=dict(size=12, color="#111111")),
                                      thickness=12, len=0.7,
                                      tickfont=dict(size=11, color="#111111"))),
            marker_line_width=0,
            text=rep_bur["Rate"].apply(lambda x: f"{x:.0f}% recurring"),
            textposition="outside",
            textfont=dict(size=11, color="#111111"),
            hovertemplate="<b>%{y}</b><br>Recurring Count: %{x}<br>%{text}<extra></extra>",
        ))
        fig_rbur.update_layout(
            **chart_base(height=420, margin=dict(l=215, r=100, t=10, b=30)),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9",
                       title=dict(text="Number of Recurring Findings", font=dict(size=14, color="#111111")),
                       tickfont=dict(size=12, color="#111111")),
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color="#111111")),
        )
        st.plotly_chart(fig_rbur, use_container_width=True)

    spacer()

    section("How Long Overdue Corrective Actions Have Been Open",
            "Each bar shows one bureau's overdue open findings, broken down by age. "
            "Darker shades signal longer-neglected issues — the most urgent CAP backlog.")

    overdue_df = df[(df["CAP Status"] == "Open") & (df["CAP Days Overdue"] > 0)].copy()

    if overdue_df.empty:
        st.info("No overdue open findings in the current selection — corrective actions are on track.")
    else:
        overdue_df["Aging Bucket"] = pd.cut(
            overdue_df["CAP Days Overdue"],
            bins=[0, 30, 60, 90, 180, float("inf")],
            labels=list(AGING_CLR.keys()),
        )
        top_ov = overdue_df.groupby("Bureau").size().nlargest(15).index.tolist()
        aging_data = (
            overdue_df[overdue_df["Bureau"].isin(top_ov)]
            .groupby(["Bureau","Aging Bucket"], observed=True)
            .size().reset_index(name="Count")
        )
        bureau_order = (aging_data.groupby("Bureau")["Count"].sum()
                        .sort_values(ascending=True).index.tolist())

        fig_aging = px.bar(
            aging_data, x="Count", y="Bureau", color="Aging Bucket",
            orientation="h", barmode="stack",
            color_discrete_map=AGING_CLR,
            category_orders={"Bureau": bureau_order, "Aging Bucket": list(AGING_CLR.keys())},
        )
        fig_aging.update_traces(marker_line_width=0)
        fig_aging.update_layout(
            **chart_base(height=max(360, len(top_ov) * 32 + 100),
                         margin=dict(l=230, r=20, t=10, b=10)),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9",
                       title=dict(text="Number of Overdue Open Findings", font=dict(size=14, color="#111111")),
                       tickfont=dict(size=12, color="#111111")),
            yaxis=dict(showgrid=False, title="", tickfont=dict(size=12, color="#111111")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                        x=1, font=dict(size=11, color="#111111"), title_text=""),
        )
        st.plotly_chart(fig_aging, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Trends
# ══════════════════════════════════════════════════════════════════════════════

with tab_trends:

    tab_intro(
        "<strong>Is the organization getting better or worse over time?</strong> "
        "These charts track whether critical issues, recurring findings, and open "
        "backlogs are increasing year over year — an early warning signal for audit planning."
    )

    section("Year-Over-Year Risk Indicators",
            "Watch for upward trends in any line — especially High-Risk and Recurring findings.")

    yearly = (
        df.groupby("Fiscal_Year")
        .agg(High_Risk=("Risk_Level",     lambda x: (x=="High").sum()),
             Recurring =("Repeat_Finding", lambda x: (x=="Yes").sum()),
             Open      =("CAP Status",         lambda x: (x=="Open").sum()))
        .reset_index()
        .melt(id_vars="Fiscal_Year", var_name="Category", value_name="Count")
    )
    yearly["Category"] = yearly["Category"].map({
        "High_Risk": "Critical (High-Risk) Findings",
        "Recurring": "Recurring Findings",
        "Open":      "Open Findings",
    })
    fig_trend = px.line(
        yearly, x="Fiscal_Year", y="Count", color="Category", markers=True,
        color_discrete_map={
            "Critical (High-Risk) Findings": "#EF4444",
            "Recurring Findings":            "#F59E0B",
            "Open Findings":                 "#3B82F6",
        },
    )
    fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig_trend.update_layout(**chart_base(height=360),
                            xaxis_title="Fiscal Year", yaxis_title="Number of Findings")
    fig_trend.update_xaxes(**AXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_trend.update_yaxes(**YAXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_trend.update_layout(legend=LEGEND_TOP)
    st.plotly_chart(fig_trend, use_container_width=True)

    spacer()

    section("Open vs Resolved Findings Each Year",
            "Green = findings that were closed. Red = findings still open. "
            "A growing red portion means the backlog is accumulating faster than it's being resolved.")

    bar_data = df.groupby(["Fiscal_Year","CAP Status"]).size().reset_index(name="Count")
    fig_bar = px.bar(bar_data, x="Fiscal_Year", y="Count", color="CAP Status",
                     barmode="stack",
                     color_discrete_map={"Open":"#EF4444","Closed":"#10B981"})
    fig_bar.update_traces(marker_line_width=0)
    fig_bar.update_layout(**chart_base(height=340, bargap=0.35),
                          xaxis_title="Fiscal Year", yaxis_title="Number of Findings")
    fig_bar.update_xaxes(**AXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_bar.update_yaxes(**YAXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_bar.update_layout(legend=LEGEND_TOP)
    st.plotly_chart(fig_bar, use_container_width=True)

    spacer()

    section("Finding Volume — Top 5 Highest-Risk Divisions",
            "Which divisions are generating the most findings, and whether that volume is growing.")

    div_scores = ranked.groupby("Division")["Risk Score"].mean().sort_values(ascending=False)
    top5_divs  = div_scores.head(5).index.tolist()
    line_data  = (
        df[df["Division"].isin(top5_divs)]
        .groupby(["Fiscal_Year","Division"]).size()
        .reset_index(name="Finding Count")
    )
    fig_line = px.line(line_data, x="Fiscal_Year", y="Finding Count",
                       color="Division", markers=True,
                       color_discrete_sequence=CHART_PALETTE)
    fig_line.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig_line.update_layout(**chart_base(height=340),
                           xaxis_title="Fiscal Year", yaxis_title="Number of Findings")
    fig_line.update_xaxes(**AXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_line.update_yaxes(**YAXIS_DEF, title_font=dict(size=14, color="#111111"))
    fig_line.update_layout(legend=LEGEND_TOP)
    st.plotly_chart(fig_line, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Issue Heat Map
# ══════════════════════════════════════════════════════════════════════════════

with tab_heat:

    tab_intro(
        "<strong>Where are issues clustering?</strong> Each cell shows how many open findings "
        "a bureau has in a given issue category. Darker cells signal concentrated risk. "
        "Bureaus with dark cells across multiple columns have broad, systemic exposure."
    )

    open_df = df[df["CAP Status"] == "Open"]
    if open_df.empty:
        st.info("No open findings in the current selection.")
    else:
        pivot = (
            open_df.groupby(["Bureau","Finding_Theme"]).size()
            .reset_index(name="Count")
            .pivot(index="Bureau", columns="Finding_Theme", values="Count")
            .fillna(0)
        )
        z_vals    = pivot.values.astype(float)
        z_plot    = np.where(z_vals == 0, np.nan, z_vals)
        text_vals = [[str(int(v)) if v > 0 else "" for v in row] for row in z_vals]

        fig_heat = go.Figure(go.Heatmap(
            z=z_plot, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            text=text_vals, texttemplate="%{text}",
            textfont=dict(size=12, color="#111111"),
            colorscale=[[0,"#FEF9C3"],[0.33,"#FDE047"],[0.66,"#F97316"],[1,"#DC2626"]],
            colorbar=dict(title=dict(text="Open<br>Findings", font=dict(size=12, color="#111111")),
                          tickfont=dict(size=11, color="#111111"), thickness=14, len=0.8),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Issue Type: %{x}<br>Open Findings: %{text}<extra></extra>",
        ))
        row_h = max(24, 560 // max(len(pivot), 1))
        fig_heat.update_layout(**chart_base(
            plot_bgcolor="#E2E8F0",
            margin=dict(l=270, r=80, t=10, b=120),
            height=max(480, len(pivot) * row_h + 160),
        ))
        fig_heat.update_xaxes(tickangle=-35, tickfont=dict(size=12, color="#111111"), showgrid=False)
        fig_heat.update_yaxes(tickfont=dict(size=12, color="#111111"), showgrid=False)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — AI Briefing
# ══════════════════════════════════════════════════════════════════════════════

with tab_ai:

    tab_intro(
        "<strong>Get a ready-to-share risk briefing in seconds.</strong> "
        "Click the button below to generate a structured narrative summarizing the top "
        "priority areas, recurring issue patterns, aging corrective actions, and specific "
        "audit planning recommendations — written at the Deputy Commissioner level."
    )

    btn_clicked = st.button(
        "Generating AI Risk Briefing…" if st.session_state.ai_running else "Generate AI Risk Briefing",
        type="primary",
        disabled=st.session_state.ai_running,
    )

    output_slot = st.empty()

    if btn_clicked:
        if ranked.empty:
            output_slot.warning("No data available to generate a briefing.")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                output_slot.error("**ANTHROPIC_API_KEY not set.**  \n"
                                  "```bash\nexport ANTHROPIC_API_KEY=sk-ant-...\n```")
            else:
                st.session_state.ai_running = True
                output_slot.markdown("""
                <div class="ai-loading-card">
                    <div class="ai-loading-spinner"></div>
                    <div>
                        <div class="ai-loading-text-primary">Generating AI Risk Briefing…</div>
                        <div class="ai-loading-text-secondary">
                            Analyzing risk patterns and drafting executive summary.<br>
                            This usually takes 10–20 seconds.
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    msg    = client.messages.create(
                        model="claude-sonnet-4-5", max_tokens=2000,
                        messages=[{"role": "user", "content": build_ai_prompt(df, ranked)}],
                    )
                    narrative = msg.content[0].text
                    output_slot.markdown(
                        f'<div class="rid-ai-box">{narrative}</div>',
                        unsafe_allow_html=True,
                    )
                except anthropic.AuthenticationError:
                    output_slot.error("Invalid API key. Check your ANTHROPIC_API_KEY.")
                except Exception as exc:
                    output_slot.error(f"API error: {exc}")
                finally:
                    st.session_state.ai_running = False


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:

    tab_intro(
        "<strong>Full findings dataset</strong> with all filters applied. "
        "Use this to look up specific findings, verify numbers, or export for further analysis. "
        f"Showing <strong>{len(df):,} findings</strong> across "
        f"<strong>{df['Bureau'].nunique()} bureaus</strong>."
    )

    st.dataframe(df, use_container_width=True, height=580)
