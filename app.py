"""
app.py  –  Sistema de Análisis de Pozos (Multi-archivo)
Streamlit web app that wraps the petrophysical ML pipeline.
"""

# ==========================================
# 1. LIBRERÍAS
# ==========================================
import io
import traceback

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==========================================
# 2. PAGE CONFIG  (must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="Análisis de Pozos",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 3. CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    /* ---------- palette ---------- */
    :root {
        --bg-dark:    #0d1117;
        --bg-card:    #161b22;
        --border:     #30363d;
        --accent:     #f0a500;
        --accent2:    #2ea043;
        --text-main:  #e6edf3;
        --text-muted: #8b949e;
        --danger:     #da3633;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: var(--bg-dark) !important;
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 1.6rem !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: var(--accent2) !important; }

    .section-header {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    .zone-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: var(--accent);
        color: #000;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: var(--accent) !important;
        color: #000 !important;
        font-weight: 700;
        border: none;
        border-radius: 6px;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #d49200 !important;
    }

    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 4. PIPELINE FUNCTIONS
# ==========================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={"GRLE": "GR", "ILD": "RT", "NPHI_LE": "NPHI"}, inplace=True)

    clips = {"GR": (0, 300), "RHOB": (1.5, 3.0), "NPHI": (0, 0.6), "RT": (0.1, 2000)}
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    if "RHOB" in df.columns:
        df["PHI"] = (2.65 - df["RHOB"]) / 2.65

    if "GR" in df.columns:
        gr_min = df["GR"].quantile(0.02)
        gr_max = df["GR"].quantile(0.98)
        if gr_max > gr_min:
            df["VSH"] = ((df["GR"] - gr_min) / (gr_max - gr_min)).clip(0, 1)
        else:
            df["VSH"] = df["GR"] / df["GR"].max()

    return df


def classify_zones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Zone"] = "Unknown"
    df["Zone_Confidence"] = 0.0

    if "PHI" in df.columns and "VSH" in df.columns:
        df.loc[(df["VSH"] < 0.35) & (df["PHI"] > 0.10), ["Zone", "Zone_Confidence"]] = ["Reservoir Clean", 0.8]
        df.loc[(df["VSH"] >= 0.35) & (df["VSH"] < 0.65) & (df["PHI"] > 0.08), ["Zone", "Zone_Confidence"]] = ["Reservoir Shaly", 0.7]
        df.loc[df["VSH"] >= 0.65, ["Zone", "Zone_Confidence"]] = ["Shale Seal", 0.85]
        df.loc[(df["PHI"] < 0.08) & (df["PHI"] > 0.04), ["Zone", "Zone_Confidence"]] = ["Tight Reservoir", 0.6]

    return df


def classify_petrophysical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "VSH" not in df.columns:
        df["VSH"] = 0.5
    if "PHI" not in df.columns:
        df["PHI"] = 0.05

    df["Score_VSH"] = np.where(df["VSH"] < 0.3, 100, np.where(df["VSH"] < 0.7, 50, 0))
    df["Score_PHI"] = np.where(df["PHI"] > 0.15, 100, np.where(df["PHI"] > 0.05, 50, 0))
    df["Score_RT"] = np.where(df["RT"] > 20, 100, np.where(df["RT"] > 5, 50, 0)) if "RT" in df.columns else 50
    if "GR" in df.columns:
        gr_norm = ((df["GR"] - 30) / (150 - 30)).clip(0, 1)
        df["Score_GR"] = (1 - gr_norm) * 100
    else:
        df["Score_GR"] = 50

    df["Petro_Score"] = (
        df["Score_VSH"] * 0.40 + df["Score_PHI"] * 0.35 +
        df["Score_RT"] * 0.15 + df["Score_GR"] * 0.10
    )
    df["Petro_Class"] = pd.cut(
        df["Petro_Score"], bins=[0, 30, 60, 100],
        labels=["Poor", "Fair", "Good"], include_lowest=True
    )

    conditions = [
        (df["VSH"] < 0.3) & (df["PHI"] > 0.15),
        (df["VSH"] >= 0.3) & (df["VSH"] < 0.65) & (df["PHI"] > 0.08),
        (df["VSH"] >= 0.65),
        (df["PHI"] <= 0.08) & (df["PHI"] > 0.03),
        (df["PHI"] <= 0.03),
    ]
    df["Lithofacies"] = np.select(
        conditions,
        ["Clean Reservoir", "Shaly Sand", "Shale", "Tight Sand", "Dense/Calcareous"],
        default="Undifferentiated",
    )
    return df


def build_features(df: pd.DataFrame):
    features = [c for c in ["RHOB", "GR", "NPHI", "RT", "PHI", "VSH"] if c in df.columns]
    if not features:
        raise ValueError("No hay features disponibles para el modelo.")
    df_model = df[["Depth", "Well"] + features].dropna(subset=features).reset_index(drop=True)
    return df_model, features


def train_model(df_model: pd.DataFrame, features: list):
    df_model = df_model.copy()
    if "RHOB" in df_model.columns:
        df_model["Target"] = (2.65 - df_model["RHOB"]) * 10
    else:
        df_model["Target"] = df_model[features].mean(axis=1)

    X = df_model[features]
    y = df_model["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    df_model["Prediction"] = model.predict(X_scaled)
    return df_model, model


def integrate_results(df: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(df_model[["Depth", "Well", "Prediction"]], on=["Depth", "Well"], how="left")
    if df["Prediction"].notna().sum() >= 3:
        try:
            df["Reservoir_Quality"] = pd.qcut(
                df["Prediction"], 3, labels=["Low", "Medium", "High"], duplicates="drop"
            )
        except Exception:
            df["Reservoir_Quality"] = "Medium"
    else:
        df["Reservoir_Quality"] = "Low"
    return df


def compare_results(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if "Reservoir_Quality" not in df.columns or "Petro_Class" not in df.columns:
        return df, 0.0

    quality_map = {"Low": 0, "Medium": 1, "High": 2}
    petro_map = {"Poor": 0, "Fair": 1, "Good": 2}
    df["ML_Quality_Num"] = pd.to_numeric(df["Reservoir_Quality"].map(quality_map), errors="coerce")
    df["Petro_Quality_Num"] = pd.to_numeric(df["Petro_Class"].map(petro_map), errors="coerce")
    df["Quality_Diff"] = df["ML_Quality_Num"] - df["Petro_Quality_Num"]
    df["Agreement"] = np.where(df["ML_Quality_Num"] == df["Petro_Quality_Num"], "Match", "Mismatch")

    valid = df["Agreement"].notna()
    agreement_rate = (df.loc[valid, "Agreement"] == "Match").mean() * 100 if valid.any() else 0.0
    return df, agreement_rate


# ==========================================
# 5. LOADING HELPERS
# ==========================================

def load_las(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    las = lasio.read(io.StringIO(raw.decode("utf-8", errors="replace")))
    df = las.df().reset_index()

    for col in ["DEPT", "MD", "DEPTH"]:
        if col in df.columns:
            df.rename(columns={col: "Depth"}, inplace=True)

    if "Depth" not in df.columns:
        raise ValueError("El archivo LAS no contiene una columna de profundidad (DEPT / MD / DEPTH).")

    df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")
    df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)
    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    for enc in ["utf-8", "latin1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    raise ValueError("No se pudo leer el CSV con ninguna combinación de separador/encoding.")


def normalize_depth(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["DEPT", "MD", "DEPTH", "Depth"]:
        if col in df.columns:
            df.rename(columns={col: "Depth"}, inplace=True)
            break
    if "Depth" not in df.columns:
        raise ValueError("No se encontró columna de profundidad (Depth / DEPT / MD).")
    df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")
    return df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)


# ==========================================
# 6. PLOTLY LOG PLOT
# ==========================================

CURVE_CONFIG = {
    "GR":   {"color": "#2ea043", "unit": "API",     "xmin": 0,   "xmax": 150, "label": "GR"},
    "RHOB": {"color": "#f0a500", "unit": "g/cc",    "xmin": 1.8, "xmax": 2.9, "label": "RHOB"},
    "NPHI": {"color": "#58a6ff", "unit": "v/v",     "xmin": 0.5, "xmax": -0.05, "label": "NPHI"},
    "RT":   {"color": "#ff7b72", "unit": "Ω·m",     "xmin": 0.2, "xmax": 200, "label": "RT (log)"},
}


def make_log_plot(df: pd.DataFrame, well_filter: str = None) -> go.Figure:
    if well_filter:
        df = df[df["Well"] == well_filter]
    
    available = [c for c in CURVE_CONFIG if c in df.columns]
    if not available:
        return None

    n = len(available)
    fig = make_subplots(
        rows=1, cols=n,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=[CURVE_CONFIG[c]["label"] for c in available],
    )

    depth = df["Depth"]

    for i, curve in enumerate(available, start=1):
        cfg = CURVE_CONFIG[curve]
        log_x = curve == "RT"
        fig.add_trace(
            go.Scatter(
                x=df[curve],
                y=depth,
                mode="lines",
                line=dict(color=cfg["color"], width=1.2),
                name=curve,
                xaxis=f"x{i}",
            ),
            row=1, col=i,
        )
        axis_key = f"xaxis{i}" if i > 1 else "xaxis"
        fig.update_layout(
            **{
                axis_key: dict(
                    title=f"{cfg['label']} ({cfg['unit']})",
                    title_font=dict(color=cfg["color"], size=11),
                    range=[cfg["xmin"], cfg["xmax"]],
                    type="log" if log_x else "linear",
                    side="top",
                    color="#8b949e",
                    gridcolor="#21262d",
                    showgrid=True,
                )
            }
        )

    fig.update_yaxes(
        autorange="reversed",
        title_text="Profundidad (m)",
        gridcolor="#21262d",
        color="#8b949e",
        tickfont=dict(size=10),
    )
    fig.update_layout(
        height=700,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=20),
    )
    return fig


def make_zone_plot(df: pd.DataFrame, well_filter: str = None) -> go.Figure:
    if well_filter:
        df = df[df["Well"] == well_filter]
    
    if "Lithofacies" not in df.columns or "Depth" not in df.columns:
        return None

    color_map = {
        "Clean Reservoir":  "#2ea043",
        "Shaly Sand":       "#f0a500",
        "Shale":            "#8b949e",
        "Tight Sand":       "#58a6ff",
        "Dense/Calcareous": "#ff7b72",
        "Undifferentiated": "#30363d",
    }

    fig = go.Figure()
    for facies, grp in df.groupby("Lithofacies", sort=False):
        fig.add_trace(
            go.Scatter(
                x=[0] * len(grp),
                y=grp["Depth"],
                mode="markers",
                marker=dict(
                    color=color_map.get(str(facies), "#30363d"),
                    size=4,
                    symbol="square",
                ),
                name=str(facies),
            )
        )

    fig.update_yaxes(autorange="reversed", title_text="Profundidad (m)", gridcolor="#21262d", color="#8b949e")
    fig.update_xaxes(visible=False)
    fig.update_layout(
        height=700,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        margin=dict(l=10, r=10, t=40, b=20),
        title=dict(text="Litofacies", font=dict(color="#f0a500", size=13)),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=10)),
    )
    return fig


# ==========================================
# 7. STREAMLIT UI
# ==========================================

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.5rem;">
        <span style="font-size:2.4rem;">🛢️</span>
        <div>
            <h1 style="margin:0;font-family:'Courier New',monospace;
                       font-size:1.7rem;color:#f0a500;letter-spacing:0.05em;">
                SISTEMA DE ANÁLISIS DE POZOS
            </h1>
            <p style="margin:0;color:#8b949e;font-size:0.85rem;">
                Petrofísica · Litofacies · Machine Learning
            </p>
        </div>
    </div>
    <hr style="border-color:#30363d;margin-top:0.5rem;margin-bottom:1.5rem;">
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">📂 Cargar Datos</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos",
        type=["las", "LAS", "csv"],
        accept_multiple_files=True,
        help="Formatos admitidos: .las / .LAS / .csv. Puedes cargar múltiples archivos.",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#8b949e;font-size:0.8rem;">'
        "Curvas que se buscarán:<br>"
        "<b>GR · RHOB · NPHI · RT</b><br>"
        "También: GRLE, ILD, NPHI_LE (se renombran automáticamente)."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#8b949e;font-size:0.75rem;">'
        "Pipeline: preproceso → clasificación de zonas → "
        "Random Forest (200 árboles) → petrofísica → concordancia ML/Petro."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Main area ────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 0;color:#30363d;">
            <div style="font-size:4rem;margin-bottom:1rem;">🪨</div>
            <p style="font-size:1.1rem;color:#8b949e;">
                Carga uno o más archivos <b style="color:#f0a500;">.las</b> o
                <b style="color:#f0a500;">.csv</b> en la barra lateral para comenzar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load files ────────────────────────────────────────────────────────────────
all_dfs = []
errors = []

with st.spinner(f"Leyendo {len(uploaded_files)} archivo(s)…"):
    for uploaded in uploaded_files:
        file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
        well_name = uploaded.name.rsplit(".", 1)[0]
        
        try:
            if file_ext == "las":
                df_temp = load_las(uploaded)
            else:
                df_temp = load_csv(uploaded)
                df_temp = normalize_depth(df_temp)
            
            df_temp["Well"] = well_name
            all_dfs.append(df_temp)
        except Exception as exc:
            errors.append(f"❌ {uploaded.name}: {exc}")

if errors:
    for err in errors:
        st.error(err)

if not all_dfs:
    st.error("No se pudo cargar ningún archivo. Revisa los errores arriba.")
    st.stop()

# Combine all DataFrames
df_raw = pd.concat(all_dfs, ignore_index=True)

st.success(f"✅ **{len(uploaded_files)} archivo(s)** cargado(s) — {len(df_raw):,} registros totales, {df_raw.shape[1]} columnas")

# ── Well selector ─────────────────────────────────────────────────────────────
wells = df_raw["Well"].unique().tolist()
selected_well = st.selectbox("🔍 Seleccionar pozo para visualización", wells)

# ── Preview ──────────────────────────────────────────────────────────────────
with st.expander("🔎 Vista previa de los datos crudos", expanded=False):
    st.dataframe(df_raw[df_raw["Well"] == selected_well].head(5), use_container_width=True)
    depth_col = df_raw[df_raw["Well"] == selected_well]["Depth"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Profundidad mín.", f"{depth_col.min():.1f} m")
    c2.metric("Profundidad máx.", f"{depth_col.max():.1f} m")
    c3.metric("Pozos cargados", str(len(wells)))

# ── Log plot ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📈 Registro de Curvas</p>', unsafe_allow_html=True)
fig_log = make_log_plot(df_raw, well_filter=selected_well)
if fig_log:
    st.plotly_chart(fig_log, use_container_width=True)
else:
    st.warning(
        "⚠️ No se encontraron curvas estándar (GR, RHOB, NPHI, RT) en el archivo. "
        "Verifica los nombres de las columnas."
    )
    st.write("Columnas detectadas:", list(df_raw.columns))

# ── Process button ────────────────────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("🚀 Procesar Datos y Ejecutar ML", type="primary", use_container_width=True)

if run_btn:
    progress = st.progress(0, text="Iniciando pipeline…")
    status = st.empty()

    try:
        status.info("⚙️ Preprocesando curvas…")
        df = preprocess(df_raw.copy())
        progress.progress(20, text="Preproceso completo")

        status.info("🗂️ Clasificando zonas geológicas…")
        df = classify_zones(df)
        progress.progress(40, text="Zonas clasificadas")

        status.info("🤖 Construyendo features y entrenando Random Forest…")
        df_model, features = build_features(df)
        df_model, model = train_model(df_model, features)
        df = integrate_results(df, df_model)
        progress.progress(70, text="ML completado")

        status.info("🪨 Clasificando litofacies y calidad petrofísica…")
        df = classify_petrophysical(df)
        progress.progress(85, text="Petrofísica lista")

        status.info("📊 Calculando concordancia…")
        df, agreement_rate = compare_results(df)
        progress.progress(100, text="¡Listo!")
        status.empty()
        progress.empty()

        st.session_state["df_result"] = df
        st.session_state["agreement_rate"] = agreement_rate
        st.session_state["features"] = features
        st.session_state["model"] = model

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Error durante el procesamiento: {exc}")
        st.code(traceback.format_exc(), language="python")

# ── Results (shown if processing was done) ─────────────────────────────────
if "df_result" in st.session_state:
    df = st.session_state["df_result"]
    agreement_rate = st.session_state["agreement_rate"]
    features = st.session_state["features"]

    st.markdown("---")
    st.markdown('<p class="section-header">📊 Resultados</p>', unsafe_allow_html=True)

    # ── Key metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Registros procesados", f"{len(df):,}")
    c2.metric("Concordancia ML / Petro", f"{agreement_rate:.1f}%",
              delta="objetivo ≥ 70%" if agreement_rate >= 70 else None)
    c3.metric("Pozos procesados", str(len(wells)))

    if "Lithofacies" in df.columns:
        top_litho = df["Lithofacies"].value_counts().idxmax()
        c4.metric("Litofacies principal", top_litho)

    if "Petro_Class" in df.columns:
        good_pct = (df["Petro_Class"] == "Good").mean() * 100
        c5.metric("Calidad Good", f"{good_pct:.1f}%")

    # ── Distribution charts ───────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        if "Lithofacies" in df.columns:
            lf_counts = df["Lithofacies"].value_counts().reset_index()
            lf_counts.columns = ["Litofacies", "Registros"]
            colors = {
                "Clean Reservoir": "#2ea043",
                "Shaly Sand": "#f0a500",
                "Shale": "#8b949e",
                "Tight Sand": "#58a6ff",
                "Dense/Calcareous": "#ff7b72",
                "Undifferentiated": "#30363d",
            }
            bar_colors = [colors.get(f, "#e6edf3") for f in lf_counts["Litofacies"]]
            fig_bar = go.Figure(
                go.Bar(
                    x=lf_counts["Litofacies"],
                    y=lf_counts["Registros"],
                    marker_color=bar_colors,
                    text=lf_counts["Registros"],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                title="Distribución de Litofacies (Todos los pozos)",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=20, r=20, t=40, b=40),
                height=320,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        if "Petro_Class" in df.columns:
            pc_counts = df["Petro_Class"].value_counts().reset_index()
            pc_counts.columns = ["Clase", "Registros"]
            pie_colors = {"Poor": "#da3633", "Fair": "#f0a500", "Good": "#2ea043"}
            fig_pie = go.Figure(
                go.Pie(
                    labels=pc_counts["Clase"],
                    values=pc_counts["Registros"],
                    marker=dict(colors=[pie_colors.get(str(c), "#8b949e") for c in pc_counts["Clase"]]),
                    hole=0.4,
                    textinfo="label+percent",
                )
            )
            fig_pie.update_layout(
                title="Calidad Petrofísica (Todos los pozos)",
                paper_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Zone + log combo ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🗂️ Perfil de Litofacies</p>', unsafe_allow_html=True)
    fig_zone = make_zone_plot(df, well_filter=selected_well)
    if fig_zone:
        col_zone, col_log = st.columns([1, 5])
        with col_zone:
            st.plotly_chart(fig_zone, use_container_width=True)
        with col_log:
            fig_log2 = make_log_plot(df, well_filter=selected_well)
            if fig_log2:
                st.plotly_chart(fig_log2, use_container_width=True)

    # ── Result table ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Tabla de Resultados</p>', unsafe_allow_html=True)

    display_cols = ["Well", "Depth"] + features + [
        c for c in ["Zone", "Lithofacies", "Petro_Class", "Reservoir_Quality", "Agreement"]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True, height=320)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar CSV de resultados (todos los pozos)",
        data=csv_bytes,
        file_name="resultados_multiples_pozos.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Concordance detail ────────────────────────────────────────────────────
    if "Agreement" in df.columns:
        with st.expander("🔍 Detalle de concordancia ML vs Petrofísica"):
            agree_counts = df["Agreement"].value_counts().reset_index()
            agree_counts.columns = ["Estado", "Registros"]
            fig_agree = go.Figure(
                go.Bar(
                    x=agree_counts["Estado"],
                    y=agree_counts["Registros"],
                    marker_color=["#2ea043" if s == "Match" else "#da3633" for s in agree_counts["Estado"]],
                    text=agree_counts["Registros"],
                    textposition="outside",
                )
            )
            fig_agree.update_layout(
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
            )
            st.plotly_chart(fig_agree, use_container_width=True)
