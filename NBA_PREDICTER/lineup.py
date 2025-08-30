import pandas as pd
import time
import os
import unicodedata
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2

TEMPORADAS = ["2022-23"]  # Solo temporada 2022-23
BLOQUE = 600
ARCHIVO = "alineaciones_2022.csv"

def normalizar_nombre(nombre):
    if not isinstance(nombre, str):
        return nombre
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii')
    return nombre.strip()

def get_last_processed_info():
    """Devuelve el último GAME_ID procesado y su fecha, o (None, None) si no hay archivo."""
    if not os.path.exists(ARCHIVO):
        return None, None
    df_existente = pd.read_csv(ARCHIVO)
    if df_existente.empty:
        return None, None
    df_existente['FECHA'] = pd.to_datetime(df_existente['FECHA'])
    ultimo = df_existente.sort_values('FECHA').iloc[-1]
    return ultimo['GAME_ID'], ultimo['FECHA']

def guardar_parcial(alineaciones):
    df_nuevo = pd.DataFrame(alineaciones)
    posiciones_validas = ['G', 'F', 'C']
    stats_cols = ['MIN', 'PTS', 'REB', 'AST', 'FGM', 'FGA']
    df_nuevo = df_nuevo[df_nuevo['START_POSITION'].isin(posiciones_validas)]
    df_nuevo = df_nuevo.dropna(subset=stats_cols)
    df_nuevo['PLAYER_NAME'] = df_nuevo['PLAYER_NAME'].apply(normalizar_nombre)
    if os.path.exists(ARCHIVO):
        df_nuevo.to_csv(ARCHIVO, mode='a', header=False, index=False)
    else:
        df_nuevo.to_csv(ARCHIVO, index=False)
    print(f"{len(df_nuevo)} filas añadidas a {ARCHIVO}")

def main():
    alineaciones = []

    last_game_id, last_fecha = get_last_processed_info()
    if last_game_id:
        print(f"Último partido procesado: {last_game_id} con fecha {last_fecha.date()}")
    else:
        print("No se encontraron datos previos procesados.")

    all_games = []

    # Filtrar solo partidos de la temporada 2022-23
    for temporada in TEMPORADAS:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=temporada,
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        )
        games = gamefinder.get_data_frames()[0]
        games = games.sort_values(by="GAME_DATE", ascending=True)
        all_games.append(games)

    df_games = pd.concat(all_games).drop_duplicates("GAME_ID").sort_values(by="GAME_DATE", ascending=True)

    # Filtrar partidos posteriores a la última fecha procesada
    if last_fecha:
        df_games = df_games[pd.to_datetime(df_games["GAME_DATE"]) > last_fecha]

    if df_games.empty:
        print("No hay nuevos partidos por procesar.")
        return

    df_games = df_games.reset_index(drop=True)

    ids_pendientes = list(df_games["GAME_ID"])
    game_dates = {row["GAME_ID"]: row["GAME_DATE"] for idx, row in df_games.iterrows()}
    temporadas_game = {row["GAME_ID"]: row["SEASON_ID"] for idx, row in df_games.iterrows()}

    print(f"Quedan {len(ids_pendientes)} partidos pendientes a procesar.")

    ids_bloque = ids_pendientes[:BLOQUE]
    print(f"Procesando bloque de {len(ids_bloque)} partidos...")

    for idx, game_id in enumerate(ids_bloque):
        temporada = temporadas_game[game_id]
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            df_box = box.get_data_frames()[0]
            starters = df_box[df_box['START_POSITION'].notna()]
            for _, row in starters.iterrows():
                alineaciones.append({
                    "TEMPORADA": temporada,
                    "GAME_ID": game_id,
                    "FECHA": game_dates[game_id],
                    "TEAM": row['TEAM_ABBREVIATION'].upper(),
                    "PLAYER_NAME": normalizar_nombre(row['PLAYER_NAME']),
                    "START_POSITION": row['START_POSITION'],
                    "MIN": row['MIN'],
                    "PTS": row['PTS'],
                    "REB": row['REB'],
                    "AST": row['AST'],
                    "FGM": row['FGM'],
                    "FGA": row['FGA'],
                    "FG_PCT": row['FG_PCT']
                })
            if idx % 50 == 0 and idx > 0:
                print(f"{idx} partidos procesados en este bloque...")
            time.sleep(0.7)
        except Exception as e:
            print(f"Error con partido {game_id}: {e}")

    if alineaciones:
        guardar_parcial(alineaciones)
    else:
        print("No hay partidos nuevos para guardar.")

if __name__ == "__main__":
    main()





