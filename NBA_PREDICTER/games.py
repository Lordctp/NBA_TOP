# games.py
from pathlib import Path
import pandas as pd
import streamlit as st

# ================== DARK THEME & STYLES ==================
def inject_dark_theme_games():
    st.markdown("""
    <style>
      :root {
        --primary:#22c55e;
        --accent:#4ade80;
        --bg1:#0b0e13;
        --bg2:#0f1420;
        --text:#e5e7eb;
        --muted:#cbd5e1;
        --border:#1f2937;
      }

      .stApp{
        background: radial-gradient(1200px 600px at 15% 10%, #0f1420 0%, #0b0e13 35%, #0b0e13 100%) !important;
        color: var(--text) !important;
      }

      /* Headings + visible labels (WHITE) */
      h1,h2,h3,h4,h5,h6 { color:#ffffff !important; }
      p, label, span, small, strong, em { color:#ffffff; }

      .curvy-title{
        font-weight:800; letter-spacing:.3px;
        background:linear-gradient(90deg, var(--accent), #60a5fa);
        -webkit-background-clip:text; background-clip:text;
        color:transparent !important; margin-bottom:.25rem;
      }
      .subtitle{ color:var(--muted) !important; margin:0 0 .25rem 0; }

      /* ====== FORM INPUTS ====== */
      .stTextInput input, .stDateInput input{
        background:#0f172a !important;
        color:#ffffff !important;
        border-radius:10px !important;
        border:1px solid #243041 !important;
      }
      .stTextInput input::placeholder, .stDateInput input::placeholder{
        color:rgba(255,255,255,0.75) !important;
      }

      /* SELECTBOX */
      .stSelectbox div[data-baseweb="select"],
      .stSelectbox div[data-baseweb="select"] [role="combobox"]{
        background:#1e293b !important;
        border:1px solid #334155 !important;
        border-radius:8px !important;
        min-height:44px;
      }
      .stSelectbox div[data-baseweb="select"] [role="combobox"] *,
      .stSelectbox div[data-baseweb="select"] [role="combobox"] span{
        color:#ffffff !important;
      }
      .stSelectbox div[data-baseweb="select"] svg{ fill:#ffffff !important; }

      body .stApp [data-baseweb="menu"],
      body .stApp [data-baseweb="popover"]{
        background:#111827 !important;
        color:#ffffff !important;
        border:1px solid #334155 !important;
      }
      body .stApp [data-baseweb="menu"] *{ color:#ffffff !important; }
      body .stApp [data-baseweb="option"]{ background:transparent !important; }
      body .stApp [data-baseweb="option"]:hover{ background:#1f2937 !important; }
      body .stApp [data-baseweb="menu"] [aria-selected="true"]{ background:#1f2937 !important; }

      /* Tables (text & headers WHITE) */
      [data-testid="stDataFrame"] * { color:#ffffff !important; }
      [data-testid="stDataFrame"] thead tr th { color:#ffffff !important; }
      .stTable thead th, .stTable tbody td { color:#ffffff !important; }

      /* Buttons */
      .stButton>button{
        background:linear-gradient(180deg, var(--primary), #16a34a);
        color:white; border:0; border-radius:12px; font-weight:700; padding:10px 14px;
        box-shadow:0 10px 25px rgba(34,197,94,.2);
      }
      .stButton>button:hover{ transform:translateY(-1px); }

      /* Team title above starters: WHITE */
      .team-title{
        color:#ffffff !important;
        font-weight:800;
        font-size:1.05rem;
        margin:6px 0 8px;
        letter-spacing:.3px;
      }

      /* Score & standings blocks - NO BOX LOOK */
      .score-card{
        display:flex; align-items:center; justify-content:center; gap:12px;
        font-weight:800; font-size:1.1rem; padding:6px 0;
      }
      .score-team{ color:#fff; opacity:1; }
      .score-points{ color:#fff; padding:.1rem .4rem; border-radius:.35rem; background:rgba(255,255,255,.12); min-width:40px; text-align:center; }

      .standings-title{ font-weight:800; margin:.25rem 0 .25rem 0; color:#ffffff !important; }
      .standings-caption{ color:#ffffff !important; margin:0 0 .5rem 0; }
      .conf-header{ color:#ffffff !important; font-weight:700; margin:.25rem 0; }

      /* Back to Home link - LEFT, no emoji */
      .back-home-wrap{ display:flex; justify-content:flex-start; }
      .back-home{
        display:inline-block; text-decoration:none; font-weight:700;
        background:linear-gradient(180deg, var(--primary), #16a34a);
        color:white !important; padding:8px 12px; border-radius:12px; border:1px solid var(--border);
        box-shadow:0 10px 25px rgba(34,197,94,.2);
      }
      .back-home:hover{ transform: translateY(-1px); }

      /* === COMPACT LAYOUT (reduce white space) === */
      h1, h2, h3, h4, h5, h6, p.subtitle { margin-top:.2rem !important; margin-bottom:.2rem !important; }
      .stMarkdown p, label { margin-top:.2rem !important; margin-bottom:.2rem !important; }
      .block-container { padding-top:.5rem !important; padding-bottom:.5rem !important; }
      section.main > div { padding-top:.15rem !important; padding-bottom:.15rem !important; }
      [data-testid="stHorizontalBlock"] > div { margin-bottom:.25rem !important; }

      /* ELIMINAR CUALQUIER RASTRO DE .glass SI EXISTE */
      .glass { background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important; margin:0 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================== LOAD & CLEAN ==================
@st.cache_data(show_spinner=False)
def load_lineups(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().upper() for c in df.columns]

    required = {"GAME_ID","FECHA","TEAM","PLAYER_NAME","START_POSITION","PTS","REB","AST","FGM","FGA","FG_PCT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {', '.join(sorted(missing))}")

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["START_POSITION"] = df["START_POSITION"].astype(str).str.upper().str.strip()

    num_cols = ["PTS","REB","AST","FGM","FGA","FG_PCT"]
    if "MIN" in df.columns: num_cols.append("MIN")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[df["FECHA"].notna() & df["TEAM"].notna()]

@st.cache_data(show_spinner=False)
def load_partidos(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    rename = {"Player":"jugador", "Opp":"oponente", "fgp":"FG_perc"}
    for k,v in rename.items():
        if k in df.columns: df.rename(columns={k:v}, inplace=True)

    df["jugador"] = df["jugador"].astype(str).str.upper().str.strip()
    if "oponente" in df.columns:
        df["oponente"] = df["oponente"].astype(str).str.upper().str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    num_cols = ["pts","reb","ast","min","stl","blk","fga","fgm","FG_perc","3pa","3pm","3pp","tov","pf","fta"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pos" in df.columns: df["pos"] = df["pos"].astype(str).str.upper().str.strip()

    return df

@st.cache_data(show_spinner=False)
def load_resultados(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_score","away_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["home_abbr","away_abbr","home_conf","away_conf","season_label"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_standings(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["abbr"] = df["abbr"].astype(str).str.upper().str.strip()
    df["conference"] = df["conference"].astype(str).str.title().str.strip()
    for c in ["wins","losses","points_for","points_against","point_diff","conf_rank","win_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["season","abbr","conference"])

@st.cache_data(show_spinner=False)
def build_game_index(df: pd.DataFrame) -> pd.DataFrame:
    teams_per_game = (
        df.groupby("GAME_ID")["TEAM"].unique().apply(lambda t: sorted([str(x) for x in t])).reset_index(name="teams")
    )
    date_per_game = df.groupby("GAME_ID")["FECHA"].min().reset_index(name="fecha")
    idx = teams_per_game.merge(date_per_game, on="GAME_ID", how="left")

    def pair_label(teams, fecha):
        t = [*teams][:2]
        while len(t) < 2: t.append("UNK")
        return f"{t[0]} vs {t[1]} — {fecha.date().isoformat() if pd.notna(fecha) else 'n/a'}"

    idx["team_a"] = idx["teams"].apply(lambda x: x[0] if len(x)>0 else "UNK")
    idx["team_b"] = idx["teams"].apply(lambda x: x[1] if len(x)>1 else "UNK")
    idx["label"] = idx.apply(lambda r: pair_label([r["team_a"], r["team_b"]], r["fecha"]), axis=1)

    idx = idx.sort_values(["fecha","GAME_ID"], ascending=[False,False]).reset_index(drop=True)
    return idx[["GAME_ID","fecha","team_a","team_b","label"]]

# ============== UTIL ==============
def format_pct_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    try:
        med = s.dropna().median()
        if pd.notna(med) and med <= 1.2:
            s = s * 100.0
        return s.round(1).astype(str)
    except Exception:
        return s.astype(str)

def season_int_to_label(season_int: int) -> str:
    """2021 -> '2021-22'"""
    if pd.isna(season_int): return "N/A"
    a = int(season_int)
    b = (a + 1) % 100
    return f"{a}-{b:02d}"

def _position_sort_key(pos: str) -> int:
    order = {"PG":1,"SG":2,"SF":3,"PF":4,"C":5}
    return order.get(str(pos).upper(), 99)

# ================== MAIN UI ==================
def render_games(base_dir: Path | None = None):
    """
    - List games (newest → oldest) paired by GAME_ID.
    - Filters by dates and team.
    - On selection: starters + same-day stats (partidos.csv).
    - Below: game result and conference standings.
    """
    inject_dark_theme_games()

    # Back to Home (top-left, sin emoji)
    st.markdown('<div class="back-home-wrap"><a class="back-home" href="/">Back to Home</a></div>', unsafe_allow_html=True)

    st.markdown('<h1 class="curvy-title">📝 Starting Lineups</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Games from most recent to oldest. Search by date or filter by team.</p>', unsafe_allow_html=True)

    # Location
    app_dir = Path(__file__).parent.resolve()
    base = base_dir if base_dir is not None else (Path("/data") if Path("/data").is_dir() else app_dir)
    csv_lineups = base / "alineaciones.csv"
    csv_partidos = base / "partidos.csv"

    # New files (balldontlie)
    bdl_dir = base / "salida_balldontlie"
    csv_resultados = bdl_dir / "resultados_partidos.csv"
    csv_standings  = bdl_dir / "standings.csv"

    if not csv_lineups.exists():
        st.error(f"Can't find {csv_lineups}. Upload 'alineaciones.csv' to {base}."); return
    if not csv_partidos.exists():
        st.error(f"Can't find {csv_partidos}. Upload 'partidos.csv' to {base}."); return
    if not csv_resultados.exists():
        st.warning(f"Can't find {csv_resultados}. The game score will be omitted.")
    if not csv_standings.exists():
        st.warning(f"Can't find {csv_standings}. The standings table will be omitted.")

    df_lineups = load_lineups(csv_lineups)
    df_games   = build_game_index(df_lineups)
    df_part    = load_partidos(csv_partidos)
    df_res     = load_resultados(csv_resultados) if csv_resultados.exists() else pd.DataFrame()
    df_std     = load_standings(csv_standings)  if csv_standings.exists()  else pd.DataFrame()

    # ---- Filters ----
    st.markdown("#### 🔎 Search by date / team")
    c1, c2, c3 = st.columns([1,1,2])

    min_dt = df_games["fecha"].min().date() if not df_games.empty else None
    max_dt = df_games["fecha"].max().date() if not df_games.empty else None
    with c1:
        start_date = st.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt, format="YYYY-MM-DD") if min_dt else None
    with c2:
        end_date = st.date_input("To", value=max_dt, min_value=min_dt, max_value=max_dt, format="YYYY-MM-DD") if max_dt else None
    with c3:
        team_query = st.text_input("Filter by team (3-letter code optional)", placeholder="e.g., BOS, LAL, DEN ...").strip().upper()

    filtered = df_games.copy()
    if start_date: filtered = filtered[filtered["fecha"] >= pd.to_datetime(start_date)]
    if end_date:   filtered = filtered[filtered["fecha"] <= pd.to_datetime(end_date)]
    if team_query:
        mask = (filtered["team_a"].str.contains(team_query, na=False)) | (filtered["team_b"].str.contains(team_query, na=False))
        filtered = filtered[mask]

    # ---- Game selector ----
    st.markdown("#### 📅 Select a game")
    options = filtered["label"].tolist()
    key_map = dict(zip(filtered["label"], filtered["GAME_ID"]))
    if not options:
        st.info("No games match the current filters.")
        return

    selected_label = st.selectbox("Games (most recent first)", options, index=0)
    selected_gid   = key_map[selected_label]
    selected_date  = filtered.loc[filtered["GAME_ID"] == selected_gid, "fecha"].iloc[0] if selected_gid in filtered["GAME_ID"].values else None

    # ---- Starters + merge with partidos.csv ----
    st.subheader(f"Starters — {selected_label}")

    game_rows = df_lineups[df_lineups["GAME_ID"] == selected_gid].copy()
    game_rows = game_rows[game_rows["START_POSITION"].astype(str).str.len() > 0]

    teams = sorted(game_rows["TEAM"].unique().tolist())
    if len(teams) < 2:
        st.warning("Did not detect two full teams for this GAME_ID. Showing available data.")

    out_cols_lineups = ["PLAYER_NAME","START_POSITION","PTS","REB","AST","FGM","FGA","FG_PCT"]
    extra_cols = ["min","stl","blk","3pa","3pm","tov","pf","fta","3pp"]

    if selected_date is not None:
        df_part_match = df_part[df_part["fecha"] == pd.to_datetime(selected_date)]
    else:
        df_part_match = df_part.iloc[0:0].copy()
    if not df_part_match.empty:
        df_part_match = df_part_match.sort_values("fecha").drop_duplicates(subset=["jugador","fecha"], keep="last")

    cols = st.columns(2)
    for i in range(2):
        with cols[i]:
            team = teams[i] if i < len(teams) else "UNK"
            st.markdown(f'<div class="team-title">{team}</div>', unsafe_allow_html=True)

            tdf = game_rows[game_rows["TEAM"] == team][out_cols_lineups].copy()

            # FG% from lineups: convert 0–1 to 0–100 if needed
            if "FG_PCT" in tdf.columns:
                tdf["FG%"] = format_pct_series(tdf["FG_PCT"])
                tdf.drop(columns=["FG_PCT"], inplace=True)

            # Merge by player + date
            tdf["jugador_key"] = tdf["PLAYER_NAME"].astype(str).str.upper().str.strip()
            merge_cols = ["jugador","fecha"] + [c for c in extra_cols if c in df_part_match.columns]
            mdf = tdf.merge(
                df_part_match[merge_cols],
                left_on=["jugador_key"],
                right_on=["jugador"],
                how="left",
                suffixes=("","_p")
            )

            # 3P% as-is
            if "3pp" in mdf.columns:
                mdf["3P%"] = mdf["3pp"].round(1).astype(str)

            # Order by position and MIN desc
            mdf["pos_key"] = mdf["START_POSITION"].apply(_position_sort_key)
            if "min" in mdf.columns:
                mdf = mdf.sort_values(["pos_key","min"], ascending=[True,False])
            else:
                mdf = mdf.sort_values(["pos_key"], ascending=[True])

            visible = ["PLAYER_NAME","START_POSITION","min","PTS","REB","AST","FGM","FGA","FG%","stl","blk","3pa","3pm","tov","pf","fta","3P%"]
            rename  = {
                "PLAYER_NAME":"Player","START_POSITION":"Pos","min":"MIN","PTS":"PTS","REB":"REB","AST":"AST",
                "FGM":"FGM","FGA":"FGA","FG%":"FG%","stl":"STL","blk":"BLK","3pa":"3PA","3pm":"3PM","tov":"TOV","pf":"PF","fta":"FTA","3P%":"3P%"
            }
            show_df = mdf[[c for c in visible if c in mdf.columns]].rename(columns=rename)
            st.dataframe(show_df, hide_index=True, use_container_width=True)

    # ---- Game result (below lineups) ----
    if not df_res.empty:
        st.markdown("#### 🏁 Final score")

        gid_str = str(selected_gid)
        res_row = df_res[df_res["game_id"] == gid_str]

        if res_row.empty and selected_date is not None and len(teams) >= 2:
            date_mask = df_res["date"] == pd.to_datetime(selected_date).normalize()
            t1, t2 = teams[0], teams[1]
            team_mask = ((df_res["home_abbr"] == t1) & (df_res["away_abbr"] == t2)) | \
                        ((df_res["home_abbr"] == t2) & (df_res["away_abbr"] == t1))
            res_row = df_res[date_mask & team_mask]

        if not res_row.empty:
            r = res_row.iloc[0]
            home = r.get("home_abbr","HOME")
            away = r.get("away_abbr","AWAY")
            hs   = int(r.get("home_score", 0)) if pd.notna(r.get("home_score")) else None
            as_  = int(r.get("away_score", 0)) if pd.notna(r.get("away_score")) else None
            st.markdown(
                f'<div class="score-card">'
                f'  <span class="score-team">{away}</span>'
                f'  <span class="score-points">{as_ if as_ is not None else "-"}</span>'
                f'  <span>—</span>'
                f'  <span class="score-points">{hs if hs is not None else "-"}</span>'
                f'  <span class="score-team">{home}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            status = str(r.get("status","")).strip()
            if status:
                st.caption(f"Status: {status}")
        else:
            st.info("No score found in 'resultados_partidos.csv' for this game.")

    # ---- Conference standings (below result) ----
    if not df_std.empty:
        seasons_sorted = sorted(df_std["season"].dropna().astype(int).unique().tolist())
        season_options = [season_int_to_label(s) for s in seasons_sorted] if seasons_sorted else ["N/A"]
        default_season = season_int_to_label(max(seasons_sorted)) if seasons_sorted else "N/A"

        season_label_sel = st.selectbox(
            "Season for standings",
            options=season_options,
            index=season_options.index(default_season) if default_season in season_options else 0,
            key="standings_season_selector"
        )
        try:
            season_int_sel = int(season_label_sel.split("-")[0])
        except Exception:
            season_int_sel = None

        st.markdown(f'<div class="standings-title">🏆 Standings {season_label_sel}</div>', unsafe_allow_html=True)
        st.markdown('<div class="standings-caption">Conference standings (Win% formatted from 0.637 → 63.7%).</div>', unsafe_allow_html=True)

        if season_int_sel is None:
            st.info("Select a valid season to view standings.")
            return

        std = df_std[df_std["season"] == season_int_sel].copy()
        if std.empty:
            st.info("No standings for the selected season.")
            return

        std["Win%"] = (std["win_pct"] * 100).round(1).astype(str) + "%"
        std.rename(columns={
            "abbr":"Team","wins":"W","losses":"L",
            "points_for":"PF","points_against":"PA","point_diff":"Diff","conf_rank":"Rank","conference":"Conf"
        }, inplace=True)

        east = std[std["Conf"] == "East"].sort_values(["Rank","Win%"], ascending=[True, False])
        west = std[std["Conf"] == "West"].sort_values(["Rank","Win%"], ascending=[True, False])

        cols = st.columns(2)
        for dfc, title, col in [(east,"Eastern Conference", cols[0]), (west,"Western Conference", cols[1])]:
            with col:
                st.markdown(f'<div class="conf-header">{title}</div>', unsafe_allow_html=True)
                show_cols = ["Rank","Team","W","L","Win%","PF","PA","Diff"]
                st.dataframe(
                    dfc[show_cols],
                    hide_index=True,
                    use_container_width=True
                )

