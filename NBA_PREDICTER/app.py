# app.py — NBAPredictor (Improved version with more professional interface)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_tags import st_tags
from datetime import datetime
from stats_avanzadas import StatsAvanzadas
from games import render_games

# ================== CONFIG & PATHS ==================
st.set_page_config(
    page_title="NBAPredictor Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).parent.resolve()
BASE_DIR = Path("/data") if Path("/data").is_dir() else APP_DIR
MODELS_DIR = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "apuestas_log_pro.csv"
HERO_IMG = BASE_DIR / "nba_predictor_banner.png"
SIMPLE_LOG_PATH = BASE_DIR / "apuestas_log_simple.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================== IMPORT YOUR LOGIC ==================
from prueba import (
    NBADataLoader, FeatureEngineer, NBAPredictor, MonteCarloSimulator,
    recomendar_seguridad
)

# ========== HELPERS / CACHE ==========
@st.cache_resource(show_spinner=True)
def load_all_data():
    dl = NBADataLoader()
    df, df_j, df_e, df_p, df_def = dl.load_all_data()
    fe = FeatureEngineer(df, df_j, df_e, df_p, df_def)
    X = fe.construir_features()
    y = fe.get_targets()
    return dl, fe, X, y

@st.cache_resource(show_spinner=True)
def load_models_only():
    import xgboost as xgb
    models = {}
    for stat in ["pts", "reb", "ast"]:
        path = MODELS_DIR / f"model_{stat}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Upload your trained models to {MODELS_DIR}.")
        m = xgb.XGBRegressor()
        m.load_model(str(path))
        models[stat] = m
    return models

@st.cache_data(show_spinner=False)
def file_exists(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False

# FIX: robust team extraction (rival vs oponente)
def get_unique_players_and_teams(df):
    players = sorted(df['jugador'].unique().tolist()) if 'jugador' in df.columns else []
    team_col = 'rival' if 'rival' in df.columns else ('oponente' if 'oponente' in df.columns else None)
    teams = sorted(df[team_col].unique().tolist()) if team_col else []
    return players, teams

# ================== DARK THEME & STYLES ==================

def inject_dark_theme():
    st.markdown(
        """
    <style>
      :root {
        --primary: #2563eb;
        --primary-focus: #1d4ed8;
        --primary-content: #ffffff;
        --secondary: #4b5563;
        --secondary-focus: #374151;
        --secondary-content: #ffffff;
        --accent: #10b981;
        --accent-focus: #059669;
        --accent-content: #ffffff;
        --neutral: #1f2937;
        --neutral-focus: #111827;
        --neutral-content: #ffffff;
        --base-100: #1a1a1a;
        --base-200: #0d0d0d;
        --base-300: #000000;
        --base-content: #ffffff;
        --info: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
      }

      .stApp {
        background: var(--base-200) !important;
        color: var(--base-content) !important;
      }

      .overlay-card {
        background: var(--base-100) !important;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        border: 1px solid var(--base-300);
      }

      h1, h2, h3, h4, h5, h6 { color: var(--base-content) !important; }

      /* Buttons (ensure visible on dark bg) */
      .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: 1px solid var(--base-300) !important;
      }
      .stButton>button[kind="primary"] {
        background: var(--primary) !important;
        color: #ffffff !important;
      }
      .stButton>button[kind="primary"]:hover { background: var(--primary-focus) !important; transform: translateY(-1px); }
      .stButton>button[kind="secondary"] {
        background: var(--base-100) !important;
        color: #ffffff !important;
      }

      /* Metrics */
      .stMetric {
        background: var(--base-100) !important;
        border-radius: 10px; padding: 16px; border: 1px solid var(--base-300);
      }
      .stMetric label { color: var(--neutral-content) !important; font-weight: 500 !important; }
      .stMetric div { color: var(--base-content) !important; font-size: 1.5rem !important; font-weight: 600 !important; }

      /* Select boxes */
      [data-baseweb="select"] { background-color: var(--base-100) !important; border-color: var(--base-300) !important; border-radius: 8px !important; }
      [data-baseweb="select"] input { color: var(--base-content) !important; }
      [data-baseweb="menu"] { background: var(--base-100) !important; border: 1px solid var(--base-300) !important; }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { background: var(--base-100) !important; border-radius: 8px !important; padding: 8px 16px !important; margin: 0 !important; }
      .stTabs [data-baseweb="tab"] p { color: #ffffff !important; }
      .stTabs [aria-selected="true"] { background: var(--primary) !important; }
      .stTabs [aria-selected="true"] p { color: #ffffff !important; }

      /* Sliders + label text */
      .stSlider [data-testid="stTickBar"] { display: none !important; }
      .stSlider label { color: #ffffff !important; } /* Estimated minutes label in white */

      /* Text inputs + labels */
      .stTextInput input, .stTextArea textarea, .stTags input { background: var(--base-100) !important; border-color: var(--base-300) !important; color: var(--base-content) !important; border-radius: 8px !important; }
      .stTextInput label, .stTextArea label, .stTags label { color: var(--base-content) !important; }

      /* Number inputs */
      .stNumberInput label { color: #ffffff !important; }
      .stNumberInput input { background: var(--base-100) !important; color: #ffffff !important; border-color: var(--base-300) !important; }

      /* Radio (Home/Away, Last result) — ensure white text */
      .stRadio > label, .stRadio div[role="radiogroup"] label { color: #ffffff !important; }

      /* Expanders: header and content both dark, all text white */
      details[open] > summary, .streamlit-expanderHeader { background: var(--base-100) !important; color: #ffffff !important; border: 1px solid var(--base-300) !important; border-radius: 8px !important; }
      .streamlit-expanderContent { background: var(--base-100) !important; color: #ffffff !important; border: 1px solid var(--base-300) !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding-top: 0.5rem !important; }
      .streamlit-expanderContent .stMetric label, .streamlit-expanderContent .stMetric div { color: #ffffff !important; }

      /* EV table (dark themed) */
      .ev-table-wrapper { overflow-x: auto; border: 1px solid var(--base-300); border-radius: 12px; margin-top: 1rem; }
      table.ev { width: 100%; border-collapse: collapse; background: var(--base-100); color: #ffffff; }
      table.ev th, table.ev td { border: 1px solid #2a2a2a; padding: 8px 10px; text-align: left; }
      table.ev th { position: sticky; top: 0; background: #111111; z-index: 1; }
      table.ev tr:nth-child(even) { background: #161616; }
    </style>
    """,
        unsafe_allow_html=True,
    )

# ================== ENHANCED COMPONENTS ==================
def player_stats_card(jugador, df):
    if not jugador or jugador.upper() not in df['jugador'].values:
        return

    # Last 5 games
    last_5 = (
        df[df['jugador'] == jugador.upper()]
        .sort_values('fecha')
        .tail(5)
        .copy()
    )

    # Handle missing columns gracefully
    for c in ['pts','reb','ast']:
        if c not in last_5.columns:
            last_5[c] = np.nan

    avg_pts = last_5['pts'].mean()
    avg_reb = last_5['reb'].mean()
    avg_ast = last_5['ast'].mean()

    with st.expander(f"📊 Last 5 games average for {jugador}", expanded=True):
        st.metric("Average points", f"{avg_pts:.1f}")
        st.metric("Average rebounds", f"{avg_reb:.1f}")
        st.metric("Average assists", f"{avg_ast:.1f}")

        fig = px.line(
            last_5,
            x='fecha',
            y=[c for c in ['pts','reb','ast'] if c in last_5.columns],
            labels={'value': 'Stat', 'variable': 'Type'},
            title="Last 5 games performance",
            markers=True
        )
        fig.update_xaxes(tickformat="%d %b %Y")
        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def simulation_results_card(results, vol, lines, cuota_map, preds):
    with st.container():
        st.subheader("📈 Results of Simulation")

        tabs = st.tabs(["📊 Summary", "📉 Points", "🏀 Rebounds", "🎯 Assists"])

        label = {"pts":"Points","reb":"Rebounds","ast":"Assists"}
        stats = ["pts", "reb", "ast"]

        # Summary tab
        with tabs[0]:
            cols = st.columns(3)
            for i, stat in enumerate(stats):
                res = results[stat]
                prob = res["probability"]
                fair_odds = (1/prob) if prob > 0 else None
                offered = cuota_map[f"{stat}_cuota"]
                ev = (prob*offered - 1) if (offered and prob>0) else None
                safety = recomendar_seguridad(prob, vol[stat])

                with cols[i]:
                    st.metric(
                        label=f"{label[stat]} - Prediction",
                        value=f"{preds[stat]:.1f}",
                        delta=f"Prob. Over: {prob*100:.1f}%"
                    )
                    with st.expander("Details", expanded=False):
                        st.write(f"**Over line:** {lines[stat]}")
                        st.write(f"**Offered odds:** {offered:.2f}")
                        if fair_odds:
                            st.write(f"**Fair odds:** {fair_odds:.2f}")
                        if ev is not None:
                            st.write(f"**Expected Value (EV):** {ev:.3f}")
                        st.write(f"**Volatility:** {vol[stat]:.2f}")
                        st.write(f"**Safety:** {safety}")

                        p5,p25,p50,p75,p95 = res["percentiles"]
                        st.caption(f"Percentiles: 5%:{p5:.1f} · 25%:{p25:.1f} · 50%:{p50:.1f} · 75%:{p75:.1f} · 95%:{p95:.1f}")

        # Detail tabs
        for i, stat in enumerate(stats, 1):
            with tabs[i]:
                res = results[stat]
                prob = res["probability"]
                percentiles = res["percentiles"]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Mean prediction", f"{preds[stat]:.1f}")
                    st.metric("Over probability", f"{prob*100:.1f}%")
                    st.metric("Volatility", f"{vol[stat]:.2f}")
                    offered = cuota_map.get(f"{stat}_cuota") or 0
                    ev_val = (prob*offered-1) if prob>0 else np.nan
                    st.metric("Expected Value", f"{ev_val:.3f}" if np.isfinite(ev_val) else "N/A")

                with col2:
                    if len(percentiles) == 5 and not any(pd.isna(percentiles)):
                        fig = px.box(
                            x=_sample_from_percentiles(*percentiles, n=5000),
                            labels={'x': label[stat]},
                            title=f"Distribution of {label[stat]}"
                        )
                        fig.add_vline(
                            x=lines[stat],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Line: {lines[stat]}"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data to plot distribution.")


def _sample_from_percentiles(p5, p25, p50, p75, p95, n=5000):
    """Generate samples from percentiles using interpolation"""
    import numpy as np
    quantiles = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    values = np.array([p5, p25, p50, p75, p95])

    ext_quantiles = np.array([0.01, *quantiles, 0.99])
    ext_values = np.array([
        max(0, p5 - (p25-p5)),
        *values,
        p95 + (p95-p75)
    ])

    uniform_samples = np.random.uniform(0.01, 0.99, n)
    samples = np.interp(uniform_samples, ext_quantiles, ext_values)
    return np.clip(samples, 0, None)

# ================== EV TABLE (NEW) ==================
def _english_date(d):
    try:
        return d.strftime("%Y-%m-%d")
    except Exception:
        return str(d)

@st.cache_data(show_spinner=False)
def _load_ev_table() -> pd.DataFrame:
    if not (SIMPLE_LOG_PATH.is_file() and SIMPLE_LOG_PATH.stat().st_size > 0):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIMPLE_LOG_PATH)
    except Exception:
        try:
            df = pd.read_csv(SIMPLE_LOG_PATH, sep=';')
        except Exception:
            return pd.DataFrame()
    return df

def _format_ev_display(df: pd.DataFrame) -> pd.DataFrame:
    # drop requested columns if present
    to_drop = [c for c in ['hora', 'acierto', 'recomendacion'] if c in df.columns]
    out = df.drop(columns=to_drop, errors='ignore').copy()
    # replace proba_XGB == 0 with a dash
    for col in out.columns:
        low = col.lower()
        if ('proba' in low and 'xgb' in low):
            try:
                out[col] = out[col].apply(lambda v: '-' if (pd.notna(v) and float(v)==0.0) else v)
            except Exception:
                pass
    return out

def render_ev_table():
    st.markdown("---")
    st.subheader("💵 Top 25 bets by Expected Value (EV)")

    df = _load_ev_table()
    if df.empty:
        st.info(f"No bets found. Upload {SIMPLE_LOG_PATH.name} to show EV table.")
        return

    # parse date
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['__date__'] = df['fecha'].dt.date
        today = datetime.now().date()
        if (df['__date__'] == today).any():
            use_date = today
        else:
            valid_dates = df['__date__'].dropna()
            use_date = valid_dates.max() if not valid_dates.empty else None
        if use_date is None:
            filtered = df.copy()
            sub = "(no valid date column, showing all)"
        else:
            filtered = df[df['__date__'] == use_date].copy()
            sub = f"(for { _english_date(use_date) })"
    else:
        filtered = df.copy()
        sub = "(no date provided, showing all)"

    ev_col = 'valor_esperado_EV' if 'valor_esperado_EV' in filtered.columns else None
    if ev_col:
        filtered = filtered.sort_values(ev_col, ascending=False)
    top = filtered.head(25).reset_index(drop=True)

    display_df = _format_ev_display(top)

    html_table = display_df.to_html(index=False, classes='ev', border=0)
    st.markdown(f"<div style='opacity:0.9; margin-bottom:0.5rem;'>Showing the 25 bets with the highest EV {sub}.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ev-table-wrapper'>{html_table}</div>", unsafe_allow_html=True)

# ================== VIEWS ==================
def render_home():
    inject_dark_theme()

    col1, col2 = st.columns([1, 2])
    with col1:
        if file_exists(HERO_IMG):
            st.image(str(HERO_IMG), use_column_width=True)
        else:
            st.title("🏀 NBAPredictor Pro")
    with col2:
        st.markdown("""
        <div style="padding: 2rem;">
            <h1 style="margin-bottom: 1rem;">Advanced NBA prediction system</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">
                Over/Under predictions with XGBoost models and Monte Carlo simulation.
                Analyze probabilities, expected value and volatility to make better decisions.
            </p>
            <button style="
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.75rem 1.25rem;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                transition: transform .15s ease;
                font-size: 1rem;
                font-weight: 500;
                margin-top: 1rem;
                cursor: pointer;
            ">
                Start analysis
            </button>
        </div>
        """, unsafe_allow_html=True)

        st.button("Start analysis", type="primary", on_click=lambda: go("prediccion"), key="hero_start")

    st.markdown("---")
    st.subheader("📊 Main features")

    cols = st.columns(3)
    features = [
        ("🔮 Accurate predictions", "XGBoost models trained with historical data"),
        ("🎲 Monte Carlo simulation", "Probability and percentile analysis"),
        ("💰 Expected value (EV)", "Calculate the expected value of your bets"),
        ("📈 Volatility analysis", "Measure the risk of each prediction"),
        ("🏆 Updated data", "Real-time information on players and teams"),
        ("📊 Full dashboard", "Professional visualization of results"),
    ]

    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
                <div style="
                    background: var(--base-100);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 1rem;
                    border: 1px solid var(--base-300);
                ">
                    <h4>{title}</h4>
                    <p style="opacity: 0.8; margin-bottom: 0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 Getting started is easy")

    steps = st.columns(3)
    with steps[0]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">1</div>
            <h4>Select a player</h4>
            <p style="opacity: 0.8;">Search by name or pick from the list</p>
        </div>
        """, unsafe_allow_html=True)

    with steps[1]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">2</div>
            <h4>Set up the game</h4>
            <p style="opacity: 0.8;">Opponent, expected minutes, context</p>
        </div>
        """, unsafe_allow_html=True)

    with steps[2]:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">3</div>
            <h4>Review the results</h4>
            <p style="opacity: 0.8;">Probabilities, EV and recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        st.button("Start analysis →", type="primary", use_container_width=True,
                  on_click=lambda: go("prediccion"), key="home_cta")
    with col_btn2:
        st.button("📈 Advanced Stats", type="secondary", use_container_width=True,
                  on_click=lambda: go("stats_avanzadas"), key="home_stats")
    with col_btn3:
        st.button("📝 Lineups", type="secondary", use_container_width=True,
                  on_click=lambda: go("alineaciones"), key="home_lineups")

    # EV table below the three options
    render_ev_table()


def render_prediccion():
    inject_dark_theme()

    # Top bar back button instead of sidebar
    top_cols = st.columns([1, 5])
    with top_cols[0]:
        st.button("← Back to start", use_container_width=True, on_click=lambda: go("home"), key="back_home_top")
    with top_cols[1]:
        st.title("📊 Analysis of prediction")
        st.caption("Complete the game context to get detailed predictions")

    with st.form("prediction_form"):
        dl, _, _, _ = load_all_data()
        players, teams = get_unique_players_and_teams(dl.df)

        default_player = st.session_state.get("quick_fill", {}).get("jugador", "")
        default_team = st.session_state.get("quick_fill", {}).get("rival", "")

        col1, col2 = st.columns(2)

        with col1:
            jugador = st_tags(
                label="Player",
                text='Type to search',
                value=[default_player] if default_player else [],
                suggestions=players,
                key="jugador_input"
            )
            jugador = jugador[0] if jugador else ""

            rival = st_tags(
                label="Opponent",
                text='Type to search',
                value=[default_team] if default_team else [],
                suggestions=teams,
                key="rival_input"
            )
            rival = rival[0] if rival else ""

            defensor = st.text_input("Defender (optional)")

            mins = st.slider(
                "Estimated minutes",
                min_value=0.0,
                max_value=48.0,
                value=34.0,
                step=0.5,
                help="Estimate the minutes the player will play"
            )

        with col2:
            st.markdown("<div style='margin-bottom: 0.5rem;'>Home/Away</div>", unsafe_allow_html=True)
            local = st.radio(
                "Home/Away",
                ["Home", "Away"],
                horizontal=True,
                label_visibility="collapsed"
            )
            local = 1 if local == "Home" else 0

            st.markdown("<div style='margin-bottom: 0.5rem;'>Team last result</div>", unsafe_allow_html=True)
            ult_res = st.radio(
                "Last result",
                ["Win", "Loss"],
                horizontal=True,
                label_visibility="collapsed"
            )
            ult_res = 1 if ult_res == "Win" else 0

            # Lines and odds in an expander (dark background, white text)
            with st.expander("📌 Betting lines and odds", expanded=True):
                pts_linea = st.number_input("Line OVER PTS", min_value=0.0, value=25.5, step=0.5)
                pts_cuota = st.number_input("Odds PTS", min_value=1.01, value=1.85, step=0.01, format="%.2f")

                reb_linea = st.number_input("Line OVER REB", min_value=0.0, value=7.5, step=0.5)
                reb_cuota = st.number_input("Odds REB", min_value=1.01, value=1.85, step=0.01, format="%.2f")

                ast_linea = st.number_input("Line OVER AST", min_value=0.0, value=8.5, step=0.5)
                ast_cuota = st.number_input("Odds AST", min_value=1.01, value=1.85, step=0.01, format="%.2f")

        submitted = st.form_submit_button("🔍 Analyzing game", type="primary", use_container_width=True)

    if not submitted:
        try:
            if "quick_fill" in st.session_state:
                player_stats_card(st.session_state.quick_fill.get("jugador", ""), dl.df)
        except Exception:
            pass
        return

    if not jugador or not rival:
        st.error("❌ You must specify at least one player and one opponent")
        return

    with st.spinner("Loading data and models..."):
        try:
            dl, fe, X, y = load_all_data()
            models = load_models_only()

            predictor = NBAPredictor(models, fe)
            input_data = {
                "jugador": jugador,
                "rival": rival,
                "defensor": defensor,
                "mins": float(mins),
                "local": 1 if local == "Home" else 0,
                "ult_resultado": 1 if ult_res == "Win" else 0,
            }

            preds, vol = predictor.predict(input_data)

            montecarlo = MonteCarloSimulator(random_state=42)
            lines = {"pts": pts_linea, "reb": reb_linea, "ast": ast_linea}
            cuotas = {"pts_cuota": pts_cuota, "reb_cuota": reb_cuota, "ast_cuota": ast_cuota}
            results = montecarlo.calculate_probabilities(preds, vol, lines)

        except Exception as e:
            st.error(f"❌ Error while predicting: {str(e)}")
            st.stop()

    st.success("✅ Analysis complete")

    try:
        player_stats_card(jugador, dl.df)
    except Exception:
        pass
    simulation_results_card(results, vol, lines, cuotas, preds)

    with st.expander("💾 Save analysis", expanded=False):
        save_name = st.text_input("Name to save (optional)")
        if st.button("Save", type="primary"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"analysis_{jugador.replace(' ', '_')}_{timestamp}.csv"

            data = {
                "fecha": [timestamp],
                "jugador": [jugador],
                "rival": [rival],
                "defensor": [defensor],
                "minutos": [mins],
                "local": [1 if local == "Home" else 0],
                "ult_resultado": [1 if ult_res == "Win" else 0],
                "nombre_guardado": [save_name if save_name else ""],
            }

            for stat in ["pts", "reb", "ast"]:
                data.update({
                    f"pred_{stat}": [preds[stat]],
                    f"vol_{stat}": [vol[stat]],
                    f"linea_{stat}": [lines[stat]],
                    f"cuota_{stat}": [cuotas.get(f"{stat}_cuota", np.nan)],
                    f"prob_over_{stat}": [results[stat].get("probability", np.nan)],
                    f"p5_{stat}": [results[stat].get("percentiles", [np.nan]*5)[0]],
                    f"p25_{stat}": [results[stat].get("percentiles", [np.nan]*5)[1]],
                    f"p50_{stat}": [results[stat].get("percentiles", [np.nan]*5)[2]],
                    f"p75_{stat}": [results[stat].get("percentiles", [np.nan]*5)[3]],
                    f"p95_{stat}": [results[stat].get("percentiles", [np.nan]*5)[4]],
                })

            df_save = pd.DataFrame(data)
            df_save.to_csv(LOG_PATH, mode='a', header=not LOG_PATH.exists(), index=False)
            st.success(f"✅ Analysis saved as: {filename}")

# ================== ROUTER ==================
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def go(page: str):
    st.session_state["page"] = page

if st.session_state["page"] == "home":
    render_home()
elif st.session_state["page"] == "prediccion":
    render_prediccion()
elif st.session_state["page"] == "stats_avanzadas":
    stats = StatsAvanzadas()
    stats.render()
elif st.session_state["page"] == "alineaciones":
    render_games()
else:
    st.session_state["page"] = "home"
    st.rerun()


