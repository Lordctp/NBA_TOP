from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import SeasonTypeAllStar
import pandas as pd

SEASON = '2024-25'

def get_all_games_nbaapi(season=SEASON):
    # Solo partidos de temporada regular (puedes cambiar a playoffs si quieres)
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=SeasonTypeAllStar.regular
    )
    results = gamefinder.get_data_frames()[0]
    return results

def extract_games_info(df):
    # Nos quedamos con los datos relevantes y eliminamos duplicados (cada partido aparece dos veces)
    df_games = df[["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"]].copy()
    # Robustez extra: Si MATCHUP tiene NaN, no lo consideres home
    df_games["home"] = df_games["MATCHUP"].str.contains("vs.", na=False)
    # HOME team rows
    home_games = df_games[df_games["home"]].copy()
    home_games = home_games.rename(columns={
        "TEAM_NAME": "home_team",
        "PTS": "home_score"
    })
    # AWAY team rows
    away_games = df_games[~df_games["home"]].copy()
    away_games = away_games.rename(columns={
        "TEAM_NAME": "visitor_team",
        "PTS": "visitor_score"
    })
    # Merge on GAME_ID and GAME_DATE
    merged = pd.merge(
        home_games[["GAME_ID", "GAME_DATE", "home_team", "home_score"]],
        away_games[["GAME_ID", "visitor_team", "visitor_score"]],
        on="GAME_ID"
    )
    # Mantén la fecha solo una vez
    merged = merged[["GAME_ID", "GAME_DATE", "home_team", "visitor_team", "home_score", "visitor_score"]]
    return merged.sort_values("GAME_DATE")

if __name__ == "__main__":
    print("Descargando partidos NBA temporada 2024-25 (puede tardar unos segundos)...")
    df_raw = get_all_games_nbaapi(SEASON)
    df_games = extract_games_info(df_raw)
    df_games.to_csv("nba_games_2024_25.csv", index=False)
    print("¡Descarga completa! Guardado como nba_games_2024_25.csv")
    print(df_games.head())


