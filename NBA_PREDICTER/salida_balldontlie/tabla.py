# -*- coding: utf-8 -*-
# Calcula standings de Regular Season por conferencia desde resultados_partidos.csv
# pip install pandas

import os
import pandas as pd
from collections import defaultdict

# === RUTAS ===
INPUT_CSV  = "resultados_partidos.csv"         # <-- tu archivo de entrada
OUTPUT_CSV = "standings.csv"                   # <-- salida

# === Mapa de conferencias (abreviaturas NBA) ===
EAST = {"ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"}
WEST = {"DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","OKC","PHX","POR","SAC","SAS","UTA"}
TEAM_CONF = {abbr: "East" for abbr in EAST} | {abbr: "West" for abbr in WEST}

def _first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro {path}")
    df = pd.read_csv(path)

    home_abbr = _first_existing(df, ["home_abbr","home_team_abbr","HOME_ABBR"])
    away_abbr = _first_existing(df, ["away_abbr","visitor_abbr","AWAY_ABBR"])
    home_pts  = _first_existing(df, ["home_score","home_pts","HOME_PTS"])
    away_pts  = _first_existing(df, ["away_score","away_pts","AWAY_PTS"])
    season    = _first_existing(df, ["season","Season","SEASON"])
    status    = _first_existing(df, ["status","game_status","Status"])

    missing = [k for k,v in {
        "home_abbr":home_abbr, "away_abbr":away_abbr,
        "home_score":home_pts, "away_score":away_pts,
        "season":season
    }.items() if v is None]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")

    out = pd.DataFrame({
        "home_abbr": df[home_abbr].astype(str),
        "away_abbr": df[away_abbr].astype(str),
        "home_pts":  pd.to_numeric(df[home_pts], errors="coerce"),
        "away_pts":  pd.to_numeric(df[away_pts], errors="coerce"),
        "season":    pd.to_numeric(df[season], errors="coerce"),
    })
    if status: out["status"] = df[status].astype(str)

    out["home_conf"] = out["home_abbr"].map(TEAM_CONF)
    out["away_conf"] = out["away_abbr"].map(TEAM_CONF)
    return out

def compute_standings(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    # Solo partidos finalizados o con marcador válido
    if "status" in games.columns:
        mask_final = games["status"].str.contains("Final", case=False, na=False) | (
            games["home_pts"].notna() & games["away_pts"].notna()
        )
        g = games[mask_final].copy()
    else:
        g = games.dropna(subset=["home_pts","away_pts"]).copy()

    g["home_pts"] = pd.to_numeric(g["home_pts"], errors="coerce").fillna(0).astype(int)
    g["away_pts"] = pd.to_numeric(g["away_pts"], errors="coerce").fillna(0).astype(int)

    result_frames = []
    for season, df_s in g.groupby("season"):
        W, L = defaultdict(int), defaultdict(int)
        PF, PA = defaultdict(int), defaultdict(int)
        CONF = {}

        for _, r in df_s.iterrows():
            ha, aa = r["home_abbr"], r["away_abbr"]
            hs, as_ = r["home_pts"], r["away_pts"]
            CONF[ha] = TEAM_CONF.get(ha)
            CONF[aa] = TEAM_CONF.get(aa)
            PF[ha] += hs; PA[ha] += as_
            PF[aa] += as_; PA[aa] += hs
            if hs > as_: W[ha] += 1; L[aa] += 1
            elif as_ > hs: W[aa] += 1; L[ha] += 1

        teams = sorted(set(df_s["home_abbr"]) | set(df_s["away_abbr"]))
        rows = []
        for t in teams:
            wins, losses = W[t], L[t]
            gp = wins + losses
            rows.append({
                "season": int(season) if pd.notna(season) else season,
                "abbr": t,
                "conference": CONF.get(t),
                "wins": wins,
                "losses": losses,
                "win_pct": round(wins/gp, 3) if gp else 0.0,
                "points_for": PF.get(t, 0),
                "points_against": PA.get(t, 0),
                "point_diff": PF.get(t, 0) - PA.get(t, 0),
            })
        st = pd.DataFrame(rows)
        if not st.empty:
            st = st.sort_values(["conference","win_pct","point_diff"],
                                ascending=[True, False, False]).reset_index(drop=True)
            st["conf_rank"] = st.groupby("conference").cumcount() + 1
            result_frames.append(st)

    return pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

def main():
    games = load_results(INPUT_CSV)
    standings = compute_standings(games)
    standings.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ standings guardado en {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

