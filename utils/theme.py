"""
Shared Streamlit theme helpers for the Solar PV app.
"""

import streamlit as st


_BASE_THEME_CSS = """
<style>
  :root {
    --app-bg: #08131f;
    --app-bg-secondary: #0d1b2a;
    --panel-bg: rgba(15, 32, 52, 0.88);
    --panel-bg-strong: #16314f;
    --panel-border: rgba(135, 170, 205, 0.22);
    --text-primary: #f5f7fb;
    --text-secondary: #bfd0e3;
    --text-muted: #94a8bc;
    --accent: #f5b942;
    --accent-strong: #ffcf6e;
    --success: #2bb673;
    --warning: #ffb454;
    --danger: #ff7b72;
    --shadow: 0 18px 45px rgba(2, 12, 24, 0.35);
    --radius: 16px;
  }

  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at top right, rgba(245, 185, 66, 0.12), transparent 24%),
      radial-gradient(circle at top left, rgba(95, 165, 255, 0.14), transparent 28%),
      linear-gradient(180deg, var(--app-bg-secondary) 0%, var(--app-bg) 100%);
    color: var(--text-primary);
  }

  .stApp {
    color: var(--text-primary);
  }

  [data-testid="stHeader"] {
    background: rgba(8, 19, 31, 0.72);
    backdrop-filter: blur(10px);
  }

  [data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, rgba(12, 30, 49, 0.98) 0%, rgba(8, 19, 31, 0.98) 100%);
    border-right: 1px solid var(--panel-border);
  }

  [data-testid="stSidebar"] * {
    color: var(--text-primary);
  }

  [data-testid="stSidebarNav"] {
    padding-top: 1rem;
  }

  [data-testid="stSidebarNav"] a {
    border-radius: 12px;
    color: var(--text-secondary);
    margin: 0.1rem 0;
  }

  [data-testid="stSidebarNav"] a:hover,
  [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(245, 185, 66, 0.14);
    color: var(--text-primary);
  }

  h1, h2, h3, h4, h5, h6,
  p, li, label, span, div,
  [data-testid="stMarkdownContainer"],
  [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"],
  .stAlert,
  .stException {
    color: var(--text-primary);
  }

  p,
  [data-testid="stCaptionContainer"],
  .stMarkdown small {
    color: var(--text-secondary);
  }

  a {
    color: var(--accent-strong);
  }

  hr,
  [data-testid="stDivider"] {
    border-color: var(--panel-border);
  }

  [data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 39, 62, 0.9), rgba(11, 25, 40, 0.94));
    border: 1px solid var(--panel-border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 0.9rem 1rem;
  }

  [data-testid="stMetricLabel"],
  [data-testid="stMetricDelta"] {
    color: var(--text-secondary);
  }

  [data-testid="stMetricValue"] {
    color: var(--accent-strong);
  }

  .stButton > button,
  .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #ffcf6e 100%);
    color: #1b2430;
    border: none;
    border-radius: 999px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    box-shadow: 0 10px 20px rgba(245, 185, 66, 0.22);
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #ffd27d 0%, #ffe2a8 100%);
    color: #111827;
  }

  .stButton > button:focus,
  .stDownloadButton > button:focus,
  .stButton > button:focus-visible,
  .stDownloadButton > button:focus-visible {
    box-shadow: 0 0 0 0.2rem rgba(245, 185, 66, 0.25);
  }

  .stTextInput > div > div,
  .stNumberInput > div > div,
  div[data-baseweb="input"] > div,
  div[data-baseweb="select"] > div,
  div[data-baseweb="textarea"] > div,
  .stFileUploader > div,
  .stDateInput > div > div,
  .stTimeInput > div > div,
  .stMultiSelect > div > div {
    background: rgba(15, 32, 52, 0.92);
    color: var(--text-primary);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
  }

  input,
  textarea {
    color: var(--text-primary) !important;
    caret-color: var(--accent-strong);
  }

  div[data-baseweb="select"] *,
  [role="listbox"] *,
  [role="option"] {
    color: var(--text-primary) !important;
  }

  [role="listbox"] {
    background: #12263d !important;
    border: 1px solid var(--panel-border) !important;
  }

  .stRadio [role="radiogroup"] label,
  .stCheckbox label,
  .stSelectbox label,
  .stMultiSelect label,
  .stSlider label,
  .stNumberInput label,
  .stFileUploader label,
  .stTextInput label,
  .stTextArea label {
    color: var(--text-primary) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
  }

  .stTabs [data-baseweb="tab"] {
    background: rgba(16, 34, 54, 0.72);
    border: 1px solid transparent;
    border-radius: 12px 12px 0 0;
    color: var(--text-secondary);
    padding: 0.6rem 1rem;
  }

  .stTabs [aria-selected="true"] {
    background: rgba(245, 185, 66, 0.12);
    border-color: rgba(245, 185, 66, 0.32);
    color: var(--text-primary);
  }

  .stAlert {
    background: rgba(15, 32, 52, 0.86);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
  }

  .stAlert [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary);
  }

  [data-testid="stDataFrame"],
  .stTable {
    background: rgba(15, 32, 52, 0.88);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
  }

  [data-testid="stDataFrame"] div,
  [data-testid="stDataFrame"] span,
  [data-testid="stDataFrame"] p,
  .stTable td,
  .stTable th {
    color: var(--text-primary) !important;
  }

  [data-testid="stDataFrame"] [role="columnheader"] {
    background: rgba(245, 185, 66, 0.1);
  }

  [data-testid="stExpander"] {
    background: rgba(15, 32, 52, 0.72);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
  }

  [data-testid="stToolbar"],
  [data-testid="stDecoration"] {
    background: transparent;
  }

  .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
  }
"""


def apply_theme(extra_css: str = "") -> None:
    """Inject a shared, high-contrast theme into the current Streamlit page."""
    st.markdown(
        _BASE_THEME_CSS + (f"\n{extra_css}\n</style>" if extra_css else "\n</style>"),
        unsafe_allow_html=True,
    )