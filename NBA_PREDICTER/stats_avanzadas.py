# =================================================================================================
# stats_avanzadas.py — Advanced NBA Stats (EN)
# -------------------------------------------------------------------------------------------------
# This Streamlit app provides several advanced NBA stat views:
#
#   1) Player vs Defender
#   2) Team vs Position  (NOW sourced from posiciones_jugador.txt; POS normalized to G/F/C; dark table)
#   3) Player vs Team    (NOW last 4 games; show all averages incl. FG% & 3P% from totals; dark table)
#   4) Player Advanced Stats (season aggregates + last 10 averages ONLY in dark table)
#   5) Team Advanced Stats (recent results + STANDINGS shown as a dark table instead of metric blocks)
#
# Key implementation changes requested:
#   - Remove the sidebar and add a top-left "Home" button (no emoji).
#   - Player vs Team: last 4 games; show ALL averages including FG% and 3P% computed from totals; dark table.
#   - Team vs Position: ignore old posiciones.txt and read posiciones_jugador.txt; map SG/SF/C → G/F/C; dark table
#     showing all fields except n_player_games.
#   - Player Stats (Last 10): show ONLY averages in a dark table; include FG% and 3P% from totals.
#   - Standings: render in a dark table (instead of metric cards).
#
# Notes:
#   - UI strings and comments are in ENGLISH (as requested).
#   - Position normalization is centralized in `sanitize_pos()` and applied on display.
#   - A local dark theme is injected via CSS; dataframes inherit a dark look.
#   - Minimal changes elsewhere; everything not mentioned remains as-is.
# =================================================================================================

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_tags import st_tags  # type-ahead chips for players/teams/defenders


# =========================================
# Utility helpers
# =========================================
def to_percent_str(x, decimals: int = 2) -> str:
    """
    Convert numeric values to a percent string with 'decimals' precision.

    Accepts:
      - floats in [0,1] -> multiplied by 100
      - floats in [0,100] -> used as-is
      - strings with or without '%' and/or comma decimal separators
      - None/NaN -> '—'
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        if isinstance(x, str):
            s = x.strip().replace("%", "").replace(",", ".")
            if s == "":
                return "—"
            x = float(s)
        if -1.0 <= x <= 1.0:
            x = x * 100.0
        return f"{x:.{decimals}f}%"
    except Exception:
        return "—"


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Sanitize a pandas Series to numeric by:
      - stripping spaces
      - removing '%' and non-numeric symbols (keeping '.', '-', scientific notation)
      - converting ',' decimal to '.'
      - coercing errors to NaN
    """
    if s.dtype.kind in "if":
        return s
    s = (
        s.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9\.\-eE]", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def sanitize_team(v):
    """Uppercase and strip a team code/name."""
    return str(v).strip().upper() if pd.notna(v) else v


def sanitize_pos(v):
    """
    Normalize a position string to a coarse bucket:
      - PG, SG, G -> G
      - SF, PF, F -> F
      - C, FC, C-F, F-C -> C
      - For any other string that starts with G/F/C, pick its first char
      - Otherwise, return the uppercased value.
    """
    if pd.isna(v):
        return v
    s = str(v).strip().upper()
    if s in {"G", "PG", "SG"}:
        return "G"
    if s in {"F", "SF", "PF"}:
        return "F"
    if s in {"C", "FC", "C-F", "F-C"}:
        return "C"
    if len(s) > 0 and s[0] in {"G", "F", "C"}:
        return s[0]
    return s


def inject_dark_theme_local():
    """
    Inject a local dark theme and high-contrast overrides for select menus/popovers,
    plus a simple "options bar" visual class to group inputs at the top of each view.
    """
    st.markdown(
        """
    <style>
      :root {
        --primary:#2563eb; --primary-focus:#1d4ed8;
        --base-100:#111318; --base-200:#0b0e13; --base-300:#171a21;
        --text:#f8fafc; --muted:#cbd5e1; --glass: rgba(255,255,255,0.04);
        --ok:#16a34a; --bad:#ef4444;
      }

      /* Base app colors */
      .stApp { background: linear-gradient(160deg, #0b0e13 0%, #151a24 100%) !important; color: var(--text) !important; }
      h1,h2,h3,h4,h5,h6, p, label, span, div, code, pre { color: var(--text) !important; }

      /* Containers / cards */
      .overlay-card {
        background: var(--glass) !important; border-radius: 16px; padding: 22px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.45); border: 1px solid #222936; backdrop-filter: blur(6px);
      }
      .option-bar {
        background: var(--glass) !important; border: 1px solid #222936; border-radius: 14px;
        padding: 12px 14px; margin-bottom: 12px;
      }

      /* Buttons */
      .stButton>button {
        border-radius: 12px !important; font-weight: 700 !important;
      }
      .stButton>button[kind="primary"], .stButton>button[data-baseweb="button"] {
        background: var(--primary) !important; color: #fff !important; border: none !important;
      }

      /* Metrics */
      .stMetric {
        background: linear-gradient(180deg, #0f1420 0%, #0b0e13 100%) !important;
        border:1px solid #1f2532; border-radius: 14px; padding: 14px 16px;
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { background:#0f1420 !important; border-radius:10px !important; }
      .stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }

      /* Data frames / alerts */
      .stDataFrame { filter: brightness(0.98); }
      .stAlert { background:#0f1420 !important; border:1px solid #1f2937 !important; }

      /* High-contrast selects / popovers (TEAM/POS inputs) */
      .stSelectbox [data-baseweb="select"],
      div[role="combobox"] {
        background:#ffffff !important; color:#111 !important; border:1px solid #d1d5db !important;
      }
      .stSelectbox [data-baseweb="select"] * { color:#111 !important; }
      .stSelectbox svg { fill:#111 !important; }
      [data-baseweb="popover"], [data-baseweb="menu"] {
        background:#ffffff !important; color:#111 !important; border:1px solid #d1d5db !important;
      }
      [data-baseweb="option"] { color:#111 !important; }
      option { color:#111 !important; background:#fff !important; }

      /* W/L pills for recent games */
      .wl-row{ display:flex; gap:8px; flex-wrap:wrap; }
      .wl-pill{
        display:inline-flex; align-items:center; gap:8px;
        padding:8px 10px; border-radius:999px; border:1px solid rgba(255,255,255,.08);
        background:rgba(255,255,255,.05);
        font-weight:700; letter-spacing:.2px;
      }
      .wl-badge{
        display:inline-flex; align-items:center; justify-content:center;
        width:22px; height:22px; border-radius:50%;
        font-weight:900; color:white;
      }
      .wl-badge.w{ background: var(--ok); }
      .wl-badge.l{ background: var(--bad); }
      .wl-text{ opacity:.95; }
      .section-title{ font-weight:800; margin:0 0 .35rem 0; }
    </style>
        """,
        unsafe_allow_html=True,
    )


def season_from_date(dt: pd.Timestamp) -> str:
    """
    Returns a season label 'YY/YY' using Aug 1 cutoff.
    Example: 2023-10-01 → '23/24'.
    """
    if pd.isna(dt):
        return ""
    year = dt.year
    start_year = year if dt.month >= 8 else year - 1
    return f"{start_year % 100:02d}/{(start_year + 1) % 100:02d}"


# =========================================
# Main app class
# =========================================
class StatsAvanzadas:
    """
    'StatsAvanzadas' class name kept to preserve compatibility with app launchers.
    """

    def __init__(self):
        self.data_dir = Path(__file__).parent.resolve()

        # Dataframes initialized to None and populated in cargar_datos()
        self.df_partidos: pd.DataFrame | None = None
        self.df_jugadores: pd.DataFrame | None = None
        self.df_equipos: pd.DataFrame | None = None
        self.df_posiciones: pd.DataFrame | None = None  # (now from posiciones_jugador.txt)
        self.df_defensores: pd.DataFrame | None = None
        self.df_res_bdl: pd.DataFrame | None = None
        self.df_std: pd.DataFrame | None = None

        # Load data on init
        self.cargar_datos()

    # -------------------------
    # Typeahead sources
    # -------------------------
    def _players_list(self) -> list[str]:
        """Return an alphabetic list of player names in UPPERCASE."""
        try:
            if (
                self.df_jugadores is not None
                and not self.df_jugadores.empty
                and "jugador" in self.df_jugadores.columns
            ):
                arr = (
                    self.df_jugadores["jugador"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                    .tolist()
                )
            else:
                arr = (
                    self.df_partidos["jugador"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                    .tolist()
                )
            return sorted(arr)
        except Exception:
            return []

    def _teams_list(self) -> list[str]:
        """Return a sorted list of team codes/names in UPPERCASE."""
        try:
            if (
                self.df_equipos is not None
                and not self.df_equipos.empty
                and "TEAM" in self.df_equipos.columns
            ):
                arr = (
                    self.df_equipos["TEAM"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                    .tolist()
                )
            else:
                arr = (
                    self.df_partidos["oponente"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                    .tolist()
                )
            return sorted(arr)
        except Exception:
            return []

    def _defenders_list(self) -> list[str]:
        """Return a sorted list of defenders (UPPERCASE) from the cross table."""
        try:
            if (
                self.df_defensores is not None
                and not self.df_defensores.empty
                and "defensor" in self.df_defensores.columns
            ):
                arr = (
                    self.df_defensores["defensor"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                    .tolist()
                )
                return sorted(arr)
            return []
        except Exception:
            return []

    # -------------------------
    # Robust readers
    # -------------------------
    def _leer_equipos_robusto(self, path: Path) -> pd.DataFrame:
        """
        Read 'equipos.txt' with an auto header fallback to 12 fixed names if needed.
        Ensures numeric columns are numeric and 'TEAM' is cleaned.
        """
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            _first = next(reader, [])
            second = next(reader, [])
        if len(second) == 12:
            names = [
                "TEAM",
                "MIN",
                "OffRtg",
                "DefRtg",
                "NetRtg",
                "REB%",
                "AST_TO",
                "Opp_PTS",
                "Opp_AST",
                "Opp_REB",
                "PACE",
                "MISC",
            ]
            df = pd.read_csv(path, header=None, names=names, skiprows=1, encoding="utf-8")
        else:
            df = pd.read_csv(path, header=0, encoding="utf-8")
            df.columns = df.columns.str.strip()
            if "TEAM" not in df.columns or df.shape[1] < 11:
                names = [
                    "TEAM",
                    "MIN",
                    "OffRtg",
                    "DefRtg",
                    "NetRtg",
                    "REB%",
                    "AST_TO",
                    "Opp_PTS",
                    "Opp_AST",
                    "Opp_REB",
                    "PACE",
                    "MISC",
                ]
                df = pd.read_csv(path, header=None, names=names, skiprows=1, encoding="utf-8")
        df["TEAM"] = df["TEAM"].apply(sanitize_team)
        for c in df.columns:
            if c != "TEAM":
                df[c] = coerce_numeric_series(df[c])
        return df

    def _leer_resultados_bdl(self, path: Path) -> pd.DataFrame:
        """Read balldontlie results CSV with consistent dtypes."""
        df = pd.read_csv(path, encoding="utf-8")
        df["game_id"] = df["game_id"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["home_score", "away_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["home_abbr", "away_abbr", "home_conf", "away_conf", "season_label", "status"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.upper().str.strip()
        return df

    def _leer_standings(self, path: Path) -> pd.DataFrame:
        """
        Read 'standings.csv' (season, abbr, conference, wins, losses, win_pct, points_for,
        points_against, point_diff, conf_rank) with normalized columns and dtypes.
        """
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = [c.strip().lower() for c in df.columns]
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["abbr"] = df["abbr"].astype(str).str.upper().str.strip()
        df["conference"] = df["conference"].astype(str).str.title().str.strip()
        for c in ["wins", "losses", "win_pct", "points_for", "points_against", "point_diff", "conf_rank"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["season", "abbr", "conference"])

    def cargar_datos(self):
        """
        Load all datasets with robust header handling, then standardize key fields.
        """
        try:
            # ===================== Games =====================
            # Player,Opp,pts,reb,ast,min,stl,blk,fga,fgm,fgp,3pa,3pm,3pp,tov,pf,fta,pos,temp,fecha,local
            self.df_partidos = pd.read_csv(self.data_dir / "partidos.csv", encoding="utf-8")
            self.df_partidos.columns = self.df_partidos.columns.str.strip()
            self.df_partidos.rename(columns={"Player": "jugador", "Opp": "oponente", "fgp": "FG_perc"}, inplace=True)
            self.df_partidos["jugador"] = self.df_partidos["jugador"].astype(str).str.upper().str.strip()
            self.df_partidos["oponente"] = self.df_partidos["oponente"].astype(str).str.upper().str.strip()
            self.df_partidos["fecha"] = pd.to_datetime(self.df_partidos["fecha"], errors="coerce")
            self.df_partidos["temporada"] = self.df_partidos["fecha"].apply(season_from_date)

            # ===================== Players (advanced) =====================
            # jugador,USG_perc,TS_perc,AST_perc,REB_perc,FG_perc
            self.df_jugadores = pd.read_csv(self.data_dir / "jugadores.txt", header=0, encoding="utf-8")
            self.df_jugadores.columns = self.df_jugadores.columns.str.strip()
            if "jugador" not in self.df_jugadores.columns:
                raise KeyError("jugadores.txt must contain column 'jugador'.")
            self.df_jugadores["jugador"] = self.df_jugadores["jugador"].astype(str).str.upper().str.strip()

            # ===================== Teams =====================
            self.df_equipos = self._leer_equipos_robusto(self.data_dir / "equipos.txt")

            # ===================== Team vs Position (NEW SOURCE) =====================
            # posiciones_jugador.txt:
            # Opp,pos,pts,reb,ast,min,stl,blk,fga,fgm,fgp,3pa,3pm,3pp,tov,pf,fta,n_player_games
            df_posj = pd.read_csv(self.data_dir / "posiciones_jugador.txt", header=0, encoding="utf-8")
            df_posj.columns = df_posj.columns.str.strip()
            # normalize names and dtypes
            df_posj = df_posj.rename(columns={"Opp": "TEAM", "pos": "POS"})
            df_posj["TEAM"] = df_posj["TEAM"].apply(sanitize_team)
            df_posj["POS"] = df_posj["POS"].apply(sanitize_pos)
            # coerce numerics
            for c in ["pts","reb","ast","min","stl","blk","fga","fgm","fgp","3pa","3pm","3pp","tov","pf","fta","n_player_games"]:
                if c in df_posj.columns:
                    df_posj[c] = coerce_numeric_series(df_posj[c])

            # group to coarse POS (G/F/C) if the file contains sub-positions; recompute FG% & 3P% from totals
            group_cols = ["TEAM", "POS"]
            sum_cols = ["fga", "fgm", "3pa", "3pm"]
            mean_cols = ["pts","reb","ast","min","stl","blk","tov","pf","fta"]
            agg = {**{c:"sum" for c in sum_cols}, **{c:"mean" for c in mean_cols}}
            dfg = df_posj.groupby(group_cols, as_index=False).agg(agg)
            # recompute percentages from totals (requested)
            dfg["FG%"] = (dfg["fgm"] / dfg["fga"]).where(dfg["fga"] > 0)
            dfg["3P%"] = (dfg["3pm"] / dfg["3pa"]).where(dfg["3pa"] > 0)
            # keep a clean display dataframe (drop source percent columns; keep sums + means)
            display_cols = ["TEAM","POS","pts","reb","ast","min","stl","blk","fga","fgm","FG%","3pa","3pm","3P%","tov","pf","fta"]
            self.df_posiciones = dfg[display_cols].copy()

            # ===================== Player–Defender cross =====================
            try:
                self.df_defensores = pd.read_csv(
                    self.data_dir / "resumen_cruce_jugador_defensor.csv", header=0, encoding="utf-8"
                )
                self.df_defensores.columns = self.df_defensores.columns.str.strip()
                self.df_defensores.rename(
                    columns={
                        "Jugador": "jugador",
                        "Defensa": "defensor",
                        "Media Min": "media_min_def",
                        "Media PTS": "media_pts_def",
                        "Media REB": "media_reb_def",
                        "Media AST": "media_ast_def",
                        "%FG Media": "fg_perc_media_def",
                    },
                    inplace=True,
                )
                self.df_defensores["jugador"] = self.df_defensores["jugador"].astype(str).str.upper().str.strip()
                self.df_defensores["defensor"] = self.df_defensores["defensor"].astype(str).str.upper().str.strip()
                for c in ["media_min_def", "media_pts_def", "media_reb_def", "media_ast_def", "fg_perc_media_def"]:
                    if c in self.df_defensores.columns:
                        self.df_defensores[c] = coerce_numeric_series(self.df_defensores[c])
            except FileNotFoundError:
                self.df_defensores = pd.DataFrame(
                    columns=["jugador", "defensor", "media_min_def", "media_pts_def", "media_reb_def", "media_ast_def", "fg_perc_media_def"]
                )

            # ===================== balldontlie outputs =====================
            bdl_dir = self.data_dir / "salida_balldontlie"
            try:
                self.df_res_bdl = self._leer_resultados_bdl(bdl_dir / "resultados_partidos.csv")
            except Exception:
                self.df_res_bdl = pd.DataFrame()
            try:
                self.df_std = self._leer_standings(bdl_dir / "standings.csv")
            except Exception:
                self.df_std = pd.DataFrame()

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

    # ==========================================================
    # Views
    # ==========================================================
    # 1) Player vs Defender
    def mostrar_stats_jugador_vs_defensa(self):
        st.subheader("📊 Player vs Defender")

        # Options bar
        st.markdown("#### Options")
        with st.container():
            ob_cols = st.columns(2)
            with ob_cols[0]:
                jugador_list = self._players_list()
                jugador_sel = st_tags(
                    label="Player",
                    text="Type to search",
                    value=[],
                    suggestions=jugador_list,
                    key="pvdef_player",
                )
                jugador = (jugador_sel[0] if jugador_sel else "").upper().strip()
            with ob_cols[1]:
                defensor_list = self._defenders_list()
                defensor_sel = st_tags(
                    label="Defender",
                    text="Type to search",
                    value=[],
                    suggestions=defensor_list,
                    key="pvdef_defender",
                )
                defensor = (defensor_sel[0] if defensor_sel else "").upper().strip()

        if not (jugador and defensor):
            st.info("Pick a player and a defender to see the cross metrics.")
            return
        if self.df_defensores is None or self.df_defensores.empty:
            st.info("No player–defender cross data available.")
            return

        mask = (self.df_defensores["jugador"] == jugador) & (self.df_defensores["defensor"] == defensor)
        stats = self.df_defensores[mask]
        if stats.empty:
            st.warning("No data found for this player–defender combination.")
            return

        row = stats.iloc[0]
        st.markdown("---")
        met_cols = st.columns(5)
        with met_cols[0]:
            st.metric("Minutes", f"{row['media_min_def']:.2f}" if pd.notna(row.get("media_min_def")) else "—")
            st.caption("Average minutes vs this defender")
        with met_cols[1]:
            st.metric("Points", f"{row['media_pts_def']:.2f}" if pd.notna(row.get("media_pts_def")) else "—")
            st.caption("Average points vs this defender")
        with met_cols[2]:
            st.metric("Rebounds", f"{row['media_reb_def']:.2f}" if pd.notna(row.get("media_reb_def")) else "—")
            st.caption("Average rebounds vs this defender")
        with met_cols[3]:
            st.metric("Assists", f"{row['media_ast_def']:.2f}" if pd.notna(row.get("media_ast_def")) else "—")
            st.caption("Average assists vs this defender")
        with met_cols[4]:
            st.metric("FG%", to_percent_str(row.get("fg_perc_media_def"), 2))
            st.caption("Field goal accuracy vs this defender")

    # 2) Team vs Position (now using posiciones_jugador.txt; dark table)
    def mostrar_stats_equipo_vs_posicion(self):
        st.subheader("🏀 Team vs Position")

        # Options bar
        st.markdown("#### Options")
        with st.container():
            ob_cols = st.columns(2)
            with ob_cols[0]:
                teams_list = self._teams_list()
                team_sel = st_tags(
                    label="Team (TEAM code)",
                    text="Type to search (e.g., BOS)",
                    value=[],
                    suggestions=teams_list,
                    key="team_vs_pos_team",
                )
                equipo = sanitize_team(team_sel[0]) if team_sel else ""
            with ob_cols[1]:
                posicion = sanitize_pos(st.radio("Position (POS)", ["G", "F", "C"], horizontal=True, key="team_vs_pos_pos"))

        if not equipo:
            st.info("Pick a team to see vs-position metrics.")
            return

        df = self.df_posiciones.copy()
        df["TEAM"] = df["TEAM"].apply(sanitize_team)
        df["POS"] = df["POS"].apply(sanitize_pos)

        stats = df[(df["TEAM"] == equipo) & (df["POS"] == posicion)]
        if stats.empty:
            st.warning("No data found for this team and position.")
            with st.expander("Show available teams/positions"):
                st.write("Teams:", ", ".join(sorted(df["TEAM"].unique())))
                st.write("Positions:", ", ".join(sorted(df["POS"].unique())))
            return

        # Show all fields except n_player_games (already excluded at load), in a DARK table
        row = stats.iloc[0].copy()
        # Format percentages nicely
        if "FG%" in stats.columns:
            row["FG%"] = to_percent_str(row["FG%"], 2)
        if "3P%" in stats.columns:
            row["3P%"] = to_percent_str(row["3P%"], 2)

        # Build a 1-row dataframe for display
        order = ["TEAM","POS","pts","reb","ast","min","stl","blk","fga","fgm","FG%","3pa","3pm","3P%","tov","pf","fta"]
        disp = pd.DataFrame([row[order].to_dict()])
        disp = disp.rename(columns={
            "TEAM":"Team","POS":"POS","pts":"PTS","reb":"REB","ast":"AST","min":"MIN","stl":"STL","blk":"BLK",
            "fga":"FGA","fgm":"FGM","3pa":"3PA","3pm":"3PM","tov":"TOV","pf":"PF","fta":"FTA"
        })
        st.markdown("---")
        st.dataframe(disp, hide_index=True, use_container_width=True)

    # 3) Player vs Team (NOW last 4; averages table incl. FG% & 3P%)
    def mostrar_stats_jugador_vs_equipo(self):
        st.subheader("🎯 Player vs Team (Last 4 games)")

        # Options bar
        st.markdown("#### Options")
        with st.container():
            ob_cols = st.columns(2)
            with ob_cols[0]:
                jugador_list = self._players_list()
                jugador_sel = st_tags(
                    label="Player",
                    text="Type to search",
                    value=[],
                    suggestions=jugador_list,
                    key="player_vs_team_player",
                )
                jugador = (jugador_sel[0] if jugador_sel else "").upper().strip()
            with ob_cols[1]:
                teams_list = self._teams_list()
                team_sel = st_tags(
                    label="Opponent team (TEAM code)",
                    text="Type to search (e.g., BOS)",
                    value=[],
                    suggestions=teams_list,
                    key="player_vs_team_team",
                )
                equipo = sanitize_team(team_sel[0]) if team_sel else ""

        if not (jugador and equipo):
            st.info("Pick a player and an opponent to see the last 4 games.")
            return

        df = self.df_partidos
        last4 = (
            df[(df["jugador"] == jugador) & (df["oponente"] == equipo)]
            .sort_values("fecha", ascending=False)
            .head(4)
            .sort_values("fecha")
        )

        if last4.empty:
            st.warning("No games found for this player–team combination.")
            return

        # Line chart for PTS/REB/AST trend (kept as before)
        fig = px.line(
            last4,
            x="fecha",
            y=["pts", "reb", "ast"],
            markers=True,
            labels={"value": "Value", "variable": "Metric", "fecha": "Date"},
            title=f"PTS/REB/AST vs {equipo} (last {len(last4)})",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        # Show the last games (unchanged)
        st.write(f"**Last {len(last4)} games of {jugador} vs {equipo}:**")
        cols_order = [
            "jugador","pts","reb","ast","min","stl","blk","fga","fgm","FG_perc","3pa","3pm","3pp","tov","pf","fta","pos","fecha",
        ]
        present = [c for c in cols_order if c in last4.columns]
        table = last4[present].copy()

        if "FG_perc" in table.columns:
            table["FG%"] = table["FG_perc"].apply(lambda v: to_percent_str(v, 2))
            table.drop(columns=["FG_perc"], inplace=True)
        if "3pp" in table.columns:
            table["3P%"] = table["3pp"].apply(lambda v: to_percent_str(v, 2))
            table.drop(columns=["3pp"], inplace=True)
        if "pos" in table.columns:
            table["pos"] = table["pos"].apply(sanitize_pos)

        rename = {
            "jugador": "Player","pts": "Points","reb": "Rebounds","ast": "Assists","min": "Minutes","stl": "Steals","blk": "Blocks",
            "fga": "FGA","fgm": "FGM","3pa": "3PA","3pm": "3PM","tov": "Turnovers","pf": "Fouls","fta": "FTA","pos": "POS","fecha": "Date",
        }
        table = table.rename(columns=rename)
        for c in ["Points","Rebounds","Assists","Minutes","Steals","Blocks","FGA","FGM","3PA","3PM","Turnovers","Fouls","FTA"]:
            if c in table.columns:
                table[c] = pd.to_numeric(table[c], errors="coerce").round(2)
        st.dataframe(table, hide_index=True, use_container_width=True)

        # === NEW: show ALL averages for last 4 (including FG% & 3P% computed from totals) in a DARK table ===
        avg_stats = {}
        num_cols = ["pts","reb","ast","min","stl","blk","tov","pf","fta","fga","fgm","3pa","3pm"]
        for c in num_cols:
            if c in last4.columns:
                avg_stats[c.upper()] = float(pd.to_numeric(last4[c], errors="coerce").mean())

        # percentages from totals:
        fga_sum = pd.to_numeric(last4.get("fga", pd.Series(dtype=float)), errors="coerce").sum()
        fgm_sum = pd.to_numeric(last4.get("fgm", pd.Series(dtype=float)), errors="coerce").sum()
        pa3_sum = pd.to_numeric(last4.get("3pa", pd.Series(dtype=float)), errors="coerce").sum()
        pm3_sum = pd.to_numeric(last4.get("3pm", pd.Series(dtype=float)), errors="coerce").sum()
        fg_pct = (fgm_sum / fga_sum) if fga_sum else None
        p3_pct = (pm3_sum / pa3_sum) if pa3_sum else None

        avg_row = {
            "PTS": round(avg_stats.get("PTS", float("nan")), 2),
            "REB": round(avg_stats.get("REB", float("nan")), 2),
            "AST": round(avg_stats.get("AST", float("nan")), 2),
            "MIN": round(avg_stats.get("MIN", float("nan")), 2),
            "STL": round(avg_stats.get("STL", float("nan")), 2),
            "BLK": round(avg_stats.get("BLK", float("nan")), 2),
            "FGA": round(avg_stats.get("FGA", float("nan")), 2),
            "FGM": round(avg_stats.get("FGM", float("nan")), 2),
            "FG%": to_percent_str(fg_pct, 2),
            "3PA": round(avg_stats.get("3PA", float("nan")), 2),
            "3PM": round(avg_stats.get("3PM", float("nan")), 2),
            "3P%": to_percent_str(p3_pct, 2),
            "TOV": round(avg_stats.get("TOV", float("nan")), 2),
            "PF":  round(avg_stats.get("PF", float("nan")), 2),
            "FTA": round(avg_stats.get("FTA", float("nan")), 2),
        }
        st.markdown("**Averages across last 4 games (full set):**")
        st.dataframe(pd.DataFrame([avg_row]), hide_index=True, use_container_width=True)

    # 4) Player Advanced Stats
    def _tabla_medias_temporada(self, df_temp: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate by season:
          - Means: PTS, REB, AST, MIN, STL, BLK, TOV, PF, FTA, FGA, 3PA, 3PM (2 decimals)
          - FG% = sum(FGM)/sum(FGA); 3P% = sum(3PM)/sum(3PA) (formatted)
        """
        if df_temp.empty:
            cols = ["Season", "Games", "PTS", "REB", "AST", "MIN", "STL", "BLK", "TOV", "PF", "FTA", "FGA", "3PA", "3PM", "FG%", "3P%"]
            return pd.DataFrame(columns=cols)

        grouped = (
            df_temp.groupby("temporada")
            .agg(
                Games=("pts", "count"),
                PTS=("pts", "mean"),
                REB=("reb", "mean"),
                AST=("ast", "mean"),
                MIN=("min", "mean"),
                STL=("stl", "mean"),
                BLK=("blk", "mean"),
                TOV=("tov", "mean"),
                PF=("pf", "mean"),
                FTA=("fta", "mean"),
                FGA_m=("fga", "mean"),
                FGA_s=("fga", "sum"),
                FGM_s=("fgm", "sum"),
                PA3_m=("3pa", "mean"),
                PA3_s=("3pa", "sum"),
                PM3_m=("3pm", "mean"),
                PM3_s=("3pm", "sum"),
            )
            .reset_index()
            .rename(columns={"temporada": "Season"})
        )

        for c in ["PTS", "REB", "AST", "MIN", "STL", "BLK", "TOV", "PF", "FTA", "FGA_m", "PA3_m", "PM3_m"]:
            if c in grouped.columns:
                grouped[c] = grouped[c].round(2)

        grouped["FG%"] = (grouped["FGM_s"] / grouped["FGA_s"]).replace([pd.NA, float("inf")], 0.0)
        grouped["3P%"] = (grouped["PM3_s"] / grouped["PA3_s"]).replace([pd.NA, float("inf")], 0.0)
        grouped["FG%"] = grouped["FG%"].apply(lambda x: to_percent_str(x, 2) if pd.notna(x) else "—")
        grouped["3P%"] = grouped["3P%"].apply(lambda x: to_percent_str(x, 2) if pd.notna(x) else "—")

        grouped = grouped.rename(columns={"FGA_m": "FGA", "PA3_m": "3PA", "PM3_m": "3PM"})
        grouped = grouped.sort_values("Season")

        show_cols = ["Season", "Games", "PTS", "REB", "AST", "MIN", "STL", "BLK", "TOV", "PF", "FTA", "FGA", "3PA", "3PM", "FG%", "3P%"]
        return grouped[show_cols]

    def mostrar_stats_avanzadas_jugador(self):
        st.subheader("🌟 Player Advanced Stats")

        # Options bar
        st.markdown("#### Options")
        with st.container():
            jugador_list = self._players_list()
            jugador_sel = st_tags(
                label="Player",
                text="Type to search",
                value=[],
                suggestions=jugador_list,
                key="adv_player",
            )
            jugador = (jugador_sel[0] if jugador_sel else "").upper().strip()

        if not jugador:
            st.info("Type a player name to see advanced stats.")
            return

        stats = self.df_jugadores[self.df_jugadores["jugador"] == jugador]
        if stats.empty:
            st.warning("No advanced data found for this player.")
            return

        row = stats.iloc[0]

        fg_txt = row.get("FG_perc") if "FG_perc" in stats.columns else None
        if pd.isna(fg_txt) or fg_txt is None:
            fg_from_games = self.df_partidos.loc[self.df_partidos["jugador"] == jugador, "FG_perc"]
            fg_txt = float(fg_from_games.mean()) if not fg_from_games.empty else None

        st.markdown("---")
        cols = st.columns(5)
        with cols[0]:
            st.metric("USG%", f"{row['USG_perc']:.2f}" if pd.notna(row.get("USG_perc")) else "—")
            st.caption("Usage rate")
        with cols[1]:
            st.metric("AST%", f"{row['AST_perc']:.2f}" if pd.notna(row.get("AST_perc")) else "—")
            st.caption("Assist rate")
        with cols[2]:
            st.metric("REB%", f"{row['REB_perc']:.2f}" if pd.notna(row.get("REB_perc")) else "—")
            st.caption("Rebound rate")
        with cols[3]:
            st.metric("FG%", to_percent_str(fg_txt, 2))
            st.caption("Field goal accuracy")
        with cols[4]:
            st.metric("TS%", f"{row['TS_perc']:.2f}" if pd.notna(row.get("TS_perc")) else "—")
            st.caption("True shooting")

        st.markdown("---")
        st.subheader("📅 Averages by season")
        df_j = self.df_partidos[self.df_partidos["jugador"] == jugador].copy()
        needed_cols = ["temporada", "pts", "reb", "ast", "min", "stl", "blk", "tov", "pf", "fta", "fga", "fgm", "3pa", "3pm"]
        df_j = df_j[[c for c in needed_cols if c in df_j.columns]]
        tabla_temp = self._tabla_medias_temporada(df_j)
        st.dataframe(tabla_temp, hide_index=True, use_container_width=True)

        # === NEW: Last 10 → ONLY averages table (dark), incl. FG% & 3P% from totals ===
        st.markdown("---")
        st.subheader("🕙 Last 10 player games — Averages only")

        last10 = (
            self.df_partidos[self.df_partidos["jugador"] == jugador]
            .sort_values("fecha", ascending=False)
            .head(10)
        )

        if last10.empty:
            st.info("The player has no recorded games yet.")
        else:
            # Count stats averages
            num_cols = ["pts","reb","ast","min","stl","blk","tov","pf","fta","fga","fgm","3pa","3pm"]
            avgs = {c.upper(): float(pd.to_numeric(last10.get(c, pd.Series(dtype=float)), errors="coerce").mean()) for c in num_cols}
            # Percentages from totals
            fga10 = pd.to_numeric(last10.get("fga", pd.Series(dtype=float)), errors="coerce").sum()
            fgm10 = pd.to_numeric(last10.get("fgm", pd.Series(dtype=float)), errors="coerce").sum()
            pa310 = pd.to_numeric(last10.get("3pa", pd.Series(dtype=float)), errors="coerce").sum()
            pm310 = pd.to_numeric(last10.get("3pm", pd.Series(dtype=float)), errors="coerce").sum()
            fg10 = (fgm10 / fga10) if fga10 else None
            p3_10 = (pm310 / pa310) if pa310 else None

            avg_row_10 = {
                "PTS": round(avgs.get("PTS", float("nan")), 2),
                "REB": round(avgs.get("REB", float("nan")), 2),
                "AST": round(avgs.get("AST", float("nan")), 2),
                "MIN": round(avgs.get("MIN", float("nan")), 2),
                "STL": round(avgs.get("STL", float("nan")), 2),
                "BLK": round(avgs.get("BLK", float("nan")), 2),
                "FGA": round(avgs.get("FGA", float("nan")), 2),
                "FGM": round(avgs.get("FGM", float("nan")), 2),
                "FG%": to_percent_str(fg10, 2),
                "3PA": round(avgs.get("3PA", float("nan")), 2),
                "3PM": round(avgs.get("3PM", float("nan")), 2),
                "3P%": to_percent_str(p3_10, 2),
                "TOV": round(avgs.get("TOV", float("nan")), 2),
                "PF":  round(avgs.get("PF", float("nan")), 2),
                "FTA": round(avgs.get("FTA", float("nan")), 2),
            }
            st.dataframe(pd.DataFrame([avg_row_10]), hide_index=True, use_container_width=True)

    # 5) Team Advanced Stats
    def _ultimos5_equipo(self, abbr: str) -> pd.DataFrame:
        """Return last 5 team games (home/away) with W/L, score, opponent."""
        if self.df_res_bdl is None or self.df_res_bdl.empty:
            return pd.DataFrame()
        d = self.df_res_bdl.copy()
        abbr = str(abbr).upper().strip()
        d = d[(d["home_abbr"] == abbr) | (d["away_abbr"] == abbr)].sort_values("date", ascending=False).head(5)
        if d.empty:
            return d
        wl, res_txt, opp = [], [], []
        for _, r in d.iterrows():
            is_home = r["home_abbr"] == abbr
            my = r["home_score"] if is_home else r["away_score"]
            op = r["away_score"] if is_home else r["home_score"]
            opp_abbr = r["away_abbr"] if is_home else r["home_abbr"]
            wl.append("W" if (pd.notna(my) and pd.notna(op) and my > op) else "L")
            res_txt.append(f"{int(my) if pd.notna(my) else '-'}–{int(op) if pd.notna(op) else '-'}")
            opp.append(opp_abbr)
        d = d.copy()
        d["WL"] = wl
        d["RESULT"] = res_txt
        d["OPP"] = opp
        return d.sort_values("date", ascending=False)

    def _standings_table(self, abbr: str):
        """
        Render standings as a dark table for the given team (latest season row).
        Replaces previous metric-block layout.
        """
        if self.df_std is None or self.df_std.empty:
            st.info("Standings data not available.")
            return
        abbr = str(abbr).upper().strip()
        s = self.df_std[self.df_std["abbr"] == abbr]
        if s.empty:
            st.info("No standings found for this team.")
            return
        season_sel = int(s["season"].max())
        r = s[s["season"] == season_sel].iloc[0]

        table = pd.DataFrame([{
            "Season": season_sel,
            "Team": abbr,
            "Conference": r.get("conference"),
            "Wins": int(r["wins"]) if pd.notna(r.get("wins")) else None,
            "Losses": int(r["losses"]) if pd.notna(r.get("losses")) else None,
            "Win%": to_percent_str(r.get("win_pct"), 2),
            "PF": round(float(r.get("points_for")), 2) if pd.notna(r.get("points_for")) else None,
            "PA": round(float(r.get("points_against")), 2) if pd.notna(r.get("points_against")) else None,
            "Diff": round(float(r.get("point_diff")), 2) if pd.notna(r.get("point_diff")) else None,
            "Conf Rank": int(r.get("conf_rank")) if pd.notna(r.get("conf_rank")) else None,
        }])
        st.markdown('<div class="section-title">🏆 Standings (latest season)</div>', unsafe_allow_html=True)
        st.dataframe(table, hide_index=True, use_container_width=True)

    def mostrar_stats_avanzadas_equipo(self):
        st.subheader("🏆 Team Advanced Stats")

        # Options bar
        st.markdown("#### Options")
        with st.container():
            teams_list = self._teams_list()
            team_sel = st_tags(
                label="Team (TEAM code, e.g., BOS)",
                text="Type to search",
                value=[],
                suggestions=teams_list,
                key="adv_team",
            )
            equipo_input = (team_sel[0] if team_sel else "").upper().strip()

        if not equipo_input:
            st.info("Type a team code to see advanced metrics.")
            return

        df = self.df_equipos.copy()
        df["TEAM"] = df["TEAM"].apply(sanitize_team)

        stats = df[df["TEAM"] == equipo_input]

        if stats.empty:
            candidates = df[df["TEAM"].str.contains(equipo_input, na=False)]
            if len(candidates) == 1:
                stats = candidates
            elif len(candidates) > 1:
                st.warning("Multiple matches. Choose one of:")
                st.write(", ".join(sorted(candidates["TEAM"].unique())))
                return
            else:
                st.warning("Team not found. Try the 3-letter code (e.g., BOS).")
                with st.expander("Show available teams"):
                    st.write(", ".join(sorted(df["TEAM"].unique())))
                return

        row = stats.iloc[0]

        st.markdown("---")
        cols = st.columns(5)
        with cols[0]:
            st.metric("OffRtg", f"{row['OffRtg']:.2f}" if pd.notna(row.get("OffRtg")) else "—")
            st.caption("Offensive rating")
        with cols[1]:
            st.metric("DefRtg", f"{row['DefRtg']:.2f}" if pd.notna(row.get("DefRtg")) else "—")
            st.caption("Defensive rating")
        with cols[2]:
            st.metric("NetRtg", f"{row['NetRtg']:.2f}" if pd.notna(row.get("NetRtg")) else "—")
            st.caption("Off - Def")
        with cols[3]:
            st.metric("PACE", f"{row['PACE']:.2f}" if pd.notna(row.get("PACE")) else "—")
            st.caption("Game pace")
        with cols[4]:
            st.metric("MIN", f"{row['MIN']:.2f}" if pd.notna(row.get("MIN")) else "—")
            st.caption("Total minutes")

        pcols = st.columns(3)
        with pcols[0]:
            st.metric("Opp REB", to_percent_str(row.get("Opp_REB"), 2) if pd.notna(row.get("Opp_REB")) else "—")
            st.caption("Opponent rebounds (%)")
        with pcols[1]:
            st.metric("Opp PTS", to_percent_str(row.get("Opp_PTS"), 2) if pd.notna(row.get("Opp_PTS")) else "—")
            st.caption("Opponent points (%)")
        with pcols[2]:
            st.metric("Opp AST", to_percent_str(row.get("Opp_AST"), 2) if pd.notna(row.get("Opp_AST")) else "—")
            st.caption("Opponent assists (%)")

        if self.df_res_bdl is not None and not self.df_res_bdl.empty:
            u5 = self._ultimos5_equipo(equipo_input)
            st.markdown('<div class="section-title">🕔 Last 5 games</div>', unsafe_allow_html=True)
            if u5.empty:
                st.info("No recent results for this team.")
            else:
                pills = []
                for _, r in u5.iterrows():
                    badge = "w" if r["WL"] == "W" else "l"
                    is_home = r["home_abbr"] == equipo_input
                    vs_txt = f"vs {r['OPP']}" if is_home else f"@ {r['OPP']}"
                    date_txt = r["date"].date().isoformat() if pd.notna(r["date"]) else ""
                    pills.append(
                        f'<span class="wl-pill"><span class="wl-badge {badge}">{r["WL"]}</span>'
                        f'<span class="wl-text">{r["RESULT"]} {vs_txt} · {date_txt}</span></span>'
                    )
                st.markdown(f'<div class="wl-row">{"".join(pills)}</div>', unsafe_allow_html=True)
        else:
            st.info("Could not find 'salida_balldontlie/resultados_partidos.csv' to show results.")

        # === NEW: standings rendered as a dark TABLE ===
        self._standings_table(equipo_input)

    # ==========================================================
    # App shell render
    # ==========================================================
    def render(self):
        inject_dark_theme_local()

        # === NEW: Top-left HOME button (no sidebar, no emoji) ===
        top_cols = st.columns([0.18, 0.82])
        with top_cols[0]:
            if st.button("Home", use_container_width=True, key="btn_home_top"):
                # If your app.py listens to session state, keep this:
                st.session_state.update({"page": "home"})
                # If using Streamlit multipage (>=1.22), you can switch pages here instead:
                # try:
                #     st.switch_page("app.py")
                # except Exception:
                #     pass

        # Title + helper text (unchanged otherwise)
        st.title("📈 Advanced NBA Stats")
        st.caption("Clean options bar, normalized positions (G/F/C), and high-contrast inputs.")

        # Tabs
        tabs = st.tabs(
            [
                "Player vs Defender",
                "Team vs Position",
                "Player vs Team",
                "Player Stats",
                "Team Stats",
            ]
        )
        with tabs[0]:
            self.mostrar_stats_jugador_vs_defensa()
        with tabs[1]:
            self.mostrar_stats_equipo_vs_posicion()
        with tabs[2]:
            self.mostrar_stats_jugador_vs_equipo()
        with tabs[3]:
            self.mostrar_stats_avanzadas_jugador()
        with tabs[4]:
            self.mostrar_stats_avanzadas_equipo()


# ==========================================================
# Entrypoint
# ==========================================================
if __name__ == "__main__":
    # Local run: streamlit run stats_avanzadas.py
    app = StatsAvanzadas()
    app.render()








