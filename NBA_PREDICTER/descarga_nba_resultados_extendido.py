# -*- coding: utf-8 -*-
"""
balldontlie (free) → genera:
  1) resultados_partidos.csv (Regular Season)
  2) jugadores.csv (bio + season averages por temporada)
  3) standings.csv (por conferencia, calculado a partir de resultados)

Robusto vs 429: respeta Retry-After y usa backoff exponencial con jitter.
"""

import os
import time
import math
import random
import requests
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datetime import date

# ===================== CONFIG ===================== #
API_KEY = "c185700f-af2b-4d72-b99f-2187a8cd15b2"  # si tienes key, ponla (free también funciona con header). Si no, déjalo "".
BASE_URL = "https://api.balldontlie.io/v1"

# Temporadas (año de inicio): 2021 -> "2021-22", etc.
START_SEASON = 2021
END_SEASON   = 2024    # inclusive (2024 => 2024-25)

# Ritmo / límites
PER_PAGE   = 100          # máximo permitido por página
SLEEP_BASE = 0.25         # espera base amable entre peticiones normales
TIMEOUT    = 30           # timeout por request
MAX_RETRIES= 8            # reintentos en 429/errores transitorios

OUTDIR = "./salida_balldontlie"
RESULTS_CSV = os.path.join(OUTDIR, "resultados_partidos.csv")
PLAYERS_CSV = os.path.join(OUTDIR, "jugadores.csv")
STAND_CSV   = os.path.join(OUTDIR, "standings.csv")
# ================================================== #

def headers():
    return {"Authorization": API_KEY} if API_KEY else {}

def season_label(y: int) -> str:
    return f"{y}-{str(y+1)[-2:]}"

# --- Mapa manual Conferencia (según NBA.com) ---
EAST = {"ATL","BOS","BKN","CHA","CHI","CLE","DET","IND","MIA","MIL","NYK","ORL","PHI","TOR","WAS"}
WEST = {"DAL","DEN","GSW","HOU","LAC","LAL","MEM","MIN","NOP","OKC","PHX","POR","SAC","SAS","UTA"}
TEAM_CONFERENCE = {abbr:"East" for abbr in EAST} | {abbr:"West" for abbr in WEST}
# (Si alguna abreviatura nueva apareciera, quedará vacía y se ignora en standings.)

# ------------- Cliente robusto (manejo 429/retries) -------------
def robust_get(url, params=None):
    """GET con manejo de 429 y backoff exponencial + jitter."""
    sleep = SLEEP_BASE
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.get(url, headers=headers(), params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(5 * attempt, 30)
                time.sleep(wait + random.uniform(0, 0.5))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            # backoff con tope
            time.sleep(min(sleep, 10) + random.uniform(0, 0.5))
            sleep *= 2
            if attempt == MAX_RETRIES:
                raise
    raise RuntimeError("Unreachable")

# -------------------- GAMES (RESULTADOS) -------------------- #
def fetch_regular_season_games(season_start_year: int) -> pd.DataFrame:
    """Descarga todos los partidos de Regular Season de una temporada (paginado con cursor)."""
    rows, cursor = [], None
    while True:
        params = {
            "seasons[]": season_start_year,
            "per_page": PER_PAGE,
            "postseason": "false",
        }
        if cursor is not None:
            params["cursor"] = cursor

        r = robust_get(f"{BASE_URL}/games", params=params)
        payload = r.json()

        for g in payload.get("data", []):
            rows.append({
                "game_id": g["id"],
                "date": g["date"],  # ISO
                "season": g["season"],
                "status": g["status"],
                "home_team_id": g["home_team"]["id"],
                "home_abbr": g["home_team"]["abbreviation"],
                "home_city": g["home_team"]["city"],
                "home_name": g["home_team"].get("full_name"),
                "home_division": g["home_team"].get("division"),
                "away_team_id": g["visitor_team"]["id"],
                "away_abbr": g["visitor_team"]["abbreviation"],
                "away_city": g["visitor_team"]["city"],
                "away_name": g["visitor_team"].get("full_name"),
                "away_division": g["visitor_team"].get("division"),
                "home_score": g["home_team_score"],
                "away_score": g["visitor_team_score"],
                "postseason": g["postseason"],
                "period": g.get("period"),
            })

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(SLEEP_BASE)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        # Añade conferencia manual por abreviatura
        df["home_conf"] = df["home_abbr"].map(TEAM_CONFERENCE)
        df["away_conf"] = df["away_abbr"].map(TEAM_CONFERENCE)
    return df

# -------------------- PLAYERS (BIO) -------------------- #
def fetch_all_players() -> pd.DataFrame:
    """Descarga TODOS los jugadores (históricos + activos) de /players."""
    players, cursor = [], None
    while True:
        params = {"per_page": PER_PAGE}
        if cursor is not None:
            params["cursor"] = cursor
        r = robust_get(f"{BASE_URL}/players", params=params)
        payload = r.json()

        for p in payload.get("data", []):
            team = p.get("team") or {}
            players.append({
                "player_id": p["id"],
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "position": p.get("position"),
                "height_feet": p.get("height_feet"),
                "height_inches": p.get("height_inches"),
                "weight_pounds": p.get("weight_pounds"),
                "team_id_current": team.get("id"),
                "team_abbr_current": team.get("abbreviation"),
                "team_city_current": team.get("city"),
                "team_name_current": team.get("full_name"),
                "team_division_current": team.get("division"),
            })

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(SLEEP_BASE)

    return pd.DataFrame(players)

# -------------------- SEASON AVERAGES -------------------- #
def fetch_season_averages_for_players(season: int, player_ids: list[int], batch_size: int = 100) -> pd.DataFrame:
    """Season averages por jugador/temporada (en lotes de 100 ids)."""
    rows = []
    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i+batch_size]
        params = [("season", season)]
        for pid in batch:
            params.append(("player_ids[]", pid))
        r = robust_get(f"{BASE_URL}/season_averages", params=params)
        for d in r.json().get("data", []):
            rows.append(d)
        time.sleep(SLEEP_BASE)
    return pd.DataFrame(rows)

# -------------------- STANDINGS (AGREGADO) -------------------- #
def compute_standings_from_games(df_games: pd.DataFrame) -> pd.DataFrame:
    """Clasificación por conferencia en RS: W, L, Win%, PF, PA, Diff + rank por conf."""
    if df_games.empty:
        return pd.DataFrame()

    w = defaultdict(int); l = defaultdict(int)
    pf = defaultdict(int); pa = defaultdict(int)
    team_name = {}; team_abbr = {}; team_conf = {}

    for _, g in df_games.iterrows():
        if g["status"] != "Final":  # solo juegos acabados
            continue
        h_id, a_id = g["home_team_id"], g["away_team_id"]
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        team_name[h_id] = g.get("home_name") or g.get("home_city")
        team_name[a_id] = g.get("away_name") or g.get("away_city")
        team_abbr[h_id] = g.get("home_abbr"); team_abbr[a_id] = g.get("away_abbr")
        team_conf[h_id] = g.get("home_conf"); team_conf[a_id] = g.get("away_conf")

        pf[h_id] += hs; pa[h_id] += as_
        pf[a_id] += as_; pa[a_id] += hs

        if hs > as_:
            w[h_id] += 1; l[a_id] += 1
        elif as_ > hs:
            w[a_id] += 1; l[h_id] += 1

    rows = []
    for tid in sorted(set(list(w.keys()) + list(l.keys()))):
        wins, losses = w[tid], l[tid]
        games = wins + losses
        win_pct = round(wins / games, 3) if games else 0.0
        rows.append({
            "team_id": tid,
            "team": team_name.get(tid),
            "abbr": team_abbr.get(tid),
            "conference": team_conf.get(tid),
            "wins": wins,
            "losses": losses,
            "win_pct": win_pct,
            "points_for": pf.get(tid, 0),
            "points_against": pa.get(tid, 0),
            "point_diff": pf.get(tid, 0) - pa.get(tid, 0),
        })

    st = pd.DataFrame(rows)
    if st.empty:
        return st
    # ordenar por conferencia y % de victoria (desc), luego diff (desc)
    st = st.sort_values(["conference", "win_pct", "point_diff"], ascending=[True, False, False]).reset_index(drop=True)
    st["conf_rank"] = st.groupby("conference").cumcount() + 1
    return st

# ---------------------------- MAIN ---------------------------- #
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    seasons = list(range(START_SEASON, END_SEASON + 1))

    # (1) Resultados de partidos
    all_games = []
    for s in seasons:
        print(f"Descargando resultados RS {season_label(s)}…")
        df = fetch_regular_season_games(s)
        df["season_label"] = season_label(s)
        all_games.append(df)
    games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    if not games.empty:
        games = games.sort_values(["season", "date", "game_id"])
    games.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"✅ Guardado: {RESULTS_CSV} ({len(games)} partidos)")

    # (2) Jugadores: bio + season averages por temporada
    print("Descargando listado de jugadores (bio)…")
    players = fetch_all_players()
    player_ids = players["player_id"].dropna().astype(int).tolist()

    avg_frames = []
    for s in seasons:
        print(f"Descargando season averages {season_label(s)}…")
        df_avg = fetch_season_averages_for_players(s, player_ids, batch_size=100)
        if not df_avg.empty:
            df_avg["season"] = s
            df_avg["season_label"] = season_label(s)
            avg_frames.append(df_avg)

    season_avg = pd.concat(avg_frames, ignore_index=True) if avg_frames else pd.DataFrame()
    if not season_avg.empty:
        jugadores = season_avg.merge(players, on="player_id", how="left")
    else:
        jugadores = players.copy()

    jugadores.to_csv(PLAYERS_CSV, index=False, encoding="utf-8")
    print(f"✅ Guardado: {PLAYERS_CSV} (filas: {len(jugadores)})")

    # (3) Standings por conferencia
    all_st = []
    if not games.empty:
        for s in seasons:
            df_s = games[games["season"] == s].copy()
            st = compute_standings_from_games(df_s)
            st["season"] = s
            st["season_label"] = season_label(s)
            all_st.append(st)
    standings = pd.concat(all_st, ignore_index=True) if all_st else pd.DataFrame()
    standings.to_csv(STAND_CSV, index=False, encoding="utf-8")
    print(f"✅ Guardado: {STAND_CSV} (filas: {len(standings)})")

if __name__ == "__main__":
    main()








