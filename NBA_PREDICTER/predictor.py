import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pickle
import os
from datetime import datetime
import csv

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== CONSTANTES ====================
COLS_PARTIDOS = [
    "jugador", "oponente", "pts", "reb", "ast", "min", "ST", "BL",
    "FGA", "FGM", "FG_perc", "TPA", "TPM", "TP_perc", "TO", "PF", "FTA", "pos", "temporada", "fecha","local"
]

COLS_JUGADORES = ["jugador", "USG_perc", "TS_perc", "AST_perc", "REB_perc"]

COLS_EQUIPOS = [
    "TEAM", "GP", "OffRtg", "DefRtg", "NetRtg", "AST_perc",
    "AST_TO", "REB_perc", "eFG_perc", "TS_perc", "PACE", "PIE", "POSS"
]
COLS_POSICIONES = ["TEAM", "pos", "ptspp", "rebpp", "asistpp"]

DEFAULT_MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

DEFAULT_VALUES = {
    'pts': 10.0,
    'reb': 4.0,
    'ast': 2.0,
    'FG_perc': 0.45,
    'vol_pts': 0.25,
    'vol_reb': 0.25,
    'vol_ast': 0.25
}

# ==================== CLASE MONTE CARLO MEJORADA ====================
class MonteCarloSimulator:
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(self.random_state)
        
    def simulate(self, mu, sigma, linea, n_simulaciones=10000):
        if mu <= 0:
            return {
                'probability': 0.0,
                'percentiles': [0]*5,
                'expected_value': mu,
                'volatility': sigma,
                'simulations': n_simulaciones,
                'distribution': 'log-normal'
            }
            
        sigma = max(min(sigma, mu * 0.75), 0.01)
        n_simulaciones = self._adjust_simulations(mu, sigma, base_simulations=n_simulaciones)
        
        sigma_log = np.sqrt(np.log(1 + (sigma**2) / (mu**2)))
        mu_log = np.log(mu) - 0.5 * sigma_log**2
        
        samples = np.random.lognormal(mean=mu_log, sigma=sigma_log, size=n_simulaciones)
        
        prob = np.mean(samples > linea)
        percentiles = np.percentile(samples, [5, 25, 50, 75, 95])
        
        return {
            'probability': prob,
            'percentiles': percentiles,
            'expected_value': mu,
            'volatility': sigma,
            'simulations': n_simulaciones,
            'distribution': 'log-normal'
        }
    
    def _adjust_simulations(self, mu, sigma, base_simulations=10000):
        if mu == 0:
            return base_simulations
        volatility_ratio = sigma / mu
        if volatility_ratio > 0.3:
            return int(base_simulations * 1.5)
        elif volatility_ratio > 0.15:
            return base_simulations
        else:
            return int(base_simulations * 0.8)
    
    def calculate_probabilities(self, preds, vol, lines):
        results = {}
        for stat in ['pts', 'reb', 'ast']:
            results[stat] = self.simulate(
                mu=preds[stat],
                sigma=preds[stat] * vol[stat],
                linea=lines[stat]
            )
        return results
# ==================== CLASE DATA LOADER ====================
import pandas as pd
import logging
import os

class NBADataLoader:
    def __init__(self):
        self.df = None
        self.df_j = None
        self.df_e = None
        self.df_p = None
        self.df_defensor = None  # Para almacenar las estadísticas del defensor

    def clean_string_columns(self, df, columns):
        """ Limpia y estandariza las columnas de texto """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
        return df

    def cargar_estadisticas_defensor(self, archivo_csv):
        """ Carga las estadísticas de los jugadores contra los defensores """
        self.df_defensor = pd.read_csv(archivo_csv)
        return self.df_defensor

    def load_all_data(self, archivo_csv):
        """ Carga todos los datos necesarios incluyendo estadísticas de enfrentamiento defensor """
        try:
            self.load_partidos()
            self.load_jugadores()
            self.load_equipos()
            self.load_posiciones()
            self.cargar_estadisticas_defensor(archivo_csv)  # Carga las estadísticas del defensor

            self.validate_data({
                'partidos': self.df,
                'jugadores': self.df_j,
                'equipos': self.df_e,
                'posiciones': self.df_p
            })

            self.add_descanso_features()  # Añade características sobre el descanso

            return self.df, self.df_j, self.df_e, self.df_p, self.df_defensor
        except Exception as e:
            logging.error("Error al cargar datos", exc_info=True)
            raise

    def validate_data(self, df_dict):
        """ Valida que los DataFrames cargados contengan todas las columnas necesarias """
        required_columns = {
            'partidos': COLS_PARTIDOS,
            'jugadores': COLS_JUGADORES,
            'equipos': COLS_EQUIPOS[:-1],  # Omitimos 'POSS'
            'posiciones': COLS_POSICIONES
        }
        for df_name, df in df_dict.items():
            if df is None:
                raise ValueError(f"DataFrame {df_name} no fue cargado correctamente")
            missing = set(required_columns[df_name]) - set(df.columns)
            if missing:
                raise ValueError(f"Faltan columnas en {df_name}: {missing}")

    def load_partidos(self):
        """ Carga los datos de los partidos """
        column_mapping = {
            'Player': 'jugador',
            'Opp': 'oponente',
            'pts': 'pts',
            'reb': 'reb',
            'ast': 'ast',
            'min': 'min',
            'stl': 'ST',
            'blk': 'BL',
            'fga': 'FGA',
            'fgm': 'FGM',
            'fgp': 'FG_perc',
            '3pa': 'TPA',
            '3pm': 'TPM',
            '3pp': 'TP_perc',
            'tov': 'TO',
            'pf': 'PF',
            'fta': 'FTA',
            'pos': 'pos',
            'temp': 'temporada',
            'local': 'local'
        }
        self.df = pd.read_csv("partidos.csv")
        self.df = self.df.rename(columns=column_mapping)
        numeric_cols = ["pts", "reb", "ast", "min", "FG_perc", "TP_perc"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], format='%Y-%m-%d', errors='coerce')
        text_cols = ["jugador", "oponente", "pos", "temporada"]
        self.df = self.clean_string_columns(self.df, text_cols)
        self.df = self.df.dropna(subset=['jugador', 'oponente', 'pts', 'reb', 'ast', 'fecha'])
        return self.df

    def load_jugadores(self):
        """ Carga los datos de los jugadores """
        self.df_j = pd.read_csv("jugadores.txt", header=None, names=COLS_JUGADORES, sep=",")
        self.df_j = self.clean_string_columns(self.df_j, ["jugador"])
        numeric_cols = ["USG_perc", "TS_perc", "AST_perc", "REB_perc"]
        for col in numeric_cols:
            self.df_j[col] = pd.to_numeric(self.df_j[col], errors='coerce')
        self.df_j = self.df_j.dropna()
        return self.df_j

    def load_equipos(self):
        """ Carga los datos de los equipos """
        self.df_e = pd.read_csv("equipos.txt", header=None, names=COLS_EQUIPOS, sep=",")
        self.df_e = self.clean_string_columns(self.df_e, ["TEAM"])
        if "POSS" in self.df_e.columns:
            self.df_e = self.df_e.drop(columns=["POSS"])
        numeric_cols = ["GP", "OffRtg", "DefRtg", "NetRtg", "AST_perc", 
                        "AST_TO", "REB_perc", "eFG_perc", "TS_perc", "PACE", "PIE"]
        for col in numeric_cols:
            self.df_e[col] = pd.to_numeric(self.df_e[col], errors='coerce')
        self.df_e = self.df_e.dropna()
        return self.df_e

    def load_posiciones(self):
        """ Carga los datos de las posiciones """
        self.df_p = pd.read_csv("posiciones.txt", header=None, names=COLS_POSICIONES, sep=",")
        self.df_p = self.clean_string_columns(self.df_p, ["TEAM", "pos"])
        numeric_cols = ["ptspp", "rebpp", "asistpp"]
        for col in numeric_cols:
            self.df_p[col] = pd.to_numeric(self.df_p[col], errors='coerce')
        self.df_p = self.df_p.dropna()
        return self.df_p

    def add_descanso_features(self):
        """ Agrega características de descanso para los jugadores y rivales """
        self.df = self.df.sort_values(['jugador', 'fecha'])
        self.df['dias_descanso_jugador'] = self.df.groupby('jugador')['fecha'].diff().dt.days.fillna(0)
        self.df = self.df.sort_values(['oponente', 'fecha'])
        self.df['dias_descanso_rival'] = self.df.groupby('oponente')['fecha'].diff().dt.days.fillna(0)
        self.df['dias_descanso_jugador'] = self.df['dias_descanso_jugador'].clip(lower=0, upper=10)
        self.df['dias_descanso_rival'] = self.df['dias_descanso_rival'].clip(lower=0, upper=10)

# ==================== CLASE FEATURE ENGINEER ====================
class FeatureEngineer:
    def __init__(self, df, df_j, df_e, df_p):
        self.df = df
        self.df_j = df_j
        self.df_e = df_e
        self.df_p = df_p
        self.le = LabelEncoder()
        self.features = None
        self._fit_encoder()

    def _fit_encoder(self):
        all_teams = pd.concat([
            self.df['oponente'],
            self.df_e['TEAM']
        ]).unique()
        self.le.fit(all_teams)
        if not os.path.exists('models'):
            os.makedirs('models')
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.le, f)

    def add_rolling_features(self):
        self.df = self.df.sort_values(['jugador', 'fecha'])
        for stat in ["pts", "reb", "ast", "FG_perc"]:
            self.df[f"{stat}_last5_vs_opp"] = self.df.groupby(['jugador', 'oponente'])[stat].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            self.df[f"{stat}_last10"] = self.df.groupby('jugador')[stat].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            self.df[f"{stat}_last_season"] = self.df.groupby(['jugador', 'temporada'])[stat].transform(
                lambda x: x.expanding(min_periods=1).mean().shift(1))
        for stat in ["pts", "reb", "ast", "FG_perc"]:
            default_value = DEFAULT_VALUES.get(stat, 0)
            for window in ['last5_vs_opp', 'last10', 'last_season']:
                col = f"{stat}_{window}"
                self.df[col] = self.df[col].fillna(default_value)
        return self

    def add_descanso_features(self):
        self.df = self.df.sort_values(['jugador', 'fecha'])
        self.df['dias_descanso_jugador'] = self.df.groupby('jugador')['fecha'].diff().dt.days.fillna(0)
        self.df['dias_descanso_jugador'] = self.df['dias_descanso_jugador'].clip(lower=0, upper=10)
        self.df = self.df.sort_values(['oponente', 'fecha'])
        self.df['dias_descanso_rival'] = self.df.groupby('oponente')['fecha'].diff().dt.days.fillna(0)
        self.df['dias_descanso_rival'] = self.df['dias_descanso_rival'].clip(lower=0, upper=10)
        return self

    def construir_features(self):
        self.add_rolling_features()
        self.add_descanso_features()
        merged = self._merge_all_data()
        merged['ultimo_resultado_equipo'] = 0
        merged['partidos_temporada'] = merged.groupby(['jugador', 'temporada'])['jugador'].transform('count')
        merged['partidos_totales'] = merged.groupby('jugador')['jugador'].transform('count')
        self.features = self._select_features(merged)
        return self.features

    def _merge_all_data(self):
        merged = self.df.merge(self.df_j, on="jugador", how="left")
        merged = merged.merge(self.df_e, left_on="oponente", right_on="TEAM", how="left")
        merged = merged.merge(
            self.df_p, left_on=["oponente", "pos"], right_on=["TEAM", "pos"], how="left"
        )
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged

    def _select_features(self, merged_df):
        merged_df['oponente_enc'] = self.le.transform(merged_df['oponente'])
        features = [
            'min', 'oponente_enc', 'USG_perc', 'TS_perc_x', 'AST_perc_x', 'REB_perc_x',
            'OffRtg', 'DefRtg', 'PACE', 'AST_perc_y', 'AST_TO', 'REB_perc_y', 'eFG_perc', 'PIE',
            'pts_last5_vs_opp', 'reb_last5_vs_opp', 'asist_last5_vs_opp', 'FG_perc_last5_vs_opp',
            'pts_last10', 'reb_last10', 'asist_last10', 'FG_perc_last10',
            'pts_last_season', 'reb_last_season', 'asist_last_season', 'FG_perc_last_season',
            'ptspp', 'rebpp', 'asistpp',
            'local', 'dias_descanso_jugador', 'dias_descanso_rival', 'ultimo_resultado_equipo',
            'partidos_temporada', 'partidos_totales'
        ]
        available_features = [f for f in features if f in merged_df.columns]
        return merged_df[available_features].copy()

    def get_targets(self):
        return {
            'pts': self.df['pts'],
            'reb': self.df['reb'],
            'ast': self.df['ast']
        }
# ==================== CLASE MODEL TRAINER ====================
class NBAModelTrainer:
    def __init__(self, X, y_dict, df_filtrado=None):
        if df_filtrado is not None:
            mask = df_filtrado.index
            X = X.loc[mask]
            y_dict = {
                'pts': y_dict['pts'].loc[mask],
                'reb': y_dict['reb'].loc[mask],
                'ast': y_dict['ast'].loc[mask],
            }
        self.X = X
        self.y_dict = y_dict
        self.models = {}
        self.best_params = {}

    def train_with_gridsearch(self):
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }
        stats = ['pts', 'reb', 'ast']

        for stat in stats:
            print(f"\n========== Entrenando y buscando hiperparámetros para {stat.upper()} ==========")
            y = self.y_dict[stat]
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y, test_size=0.2, random_state=42
            )
            xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            grid = GridSearchCV(
                xgb_reg,
                param_grid,
                scoring='neg_mean_squared_error',
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            self.models[stat] = best_model
            self.best_params[stat] = grid.best_params_
            print(f"Mejores hiperparámetros para {stat}: {grid.best_params_}")

            # Evaluación
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"R²: {r2:.3f}")
            print(f"MSE: {mse:.2f}")
            print(f"MAE: {mae:.2f}")

            # Features más importantes
            importance = best_model.feature_importances_
            features = self.X.columns
            importance_df = pd.DataFrame({'feature': features, 'importance': importance})
            importance_df = importance_df.sort_values('importance', ascending=False)
            print("\nTop 7 características importantes:")
            print(importance_df.head(7).to_string(index=False))

            # Guardar modelo
            if not os.path.exists('models'):
                os.makedirs('models')
            best_model.save_model(f'models/model_{stat}.json')

        print("\n¡Modelos entrenados y guardados exitosamente!")
        return self.models
# ==================== CLASE PREDICTOR ====================
class NBAPredictor:
    def __init__(self, models, feature_engineer):
        self.models = models
        self.feature_engineer = feature_engineer
        self.df = feature_engineer.df

    def predict(self, input_data):
        try:
            features = self._prepare_features(input_data)
            feature_order = list(self.feature_engineer.features.columns)
            features = features[feature_order]
            preds = self._make_predictions(features)
            vol = self._calculate_volatility(input_data['jugador'])
            return preds, vol
        except Exception as e:
            self._handle_error(e)
            raise

    def _prepare_features(self, input_data):
        jugador = input_data['jugador'].upper().strip()
        rival = input_data['rival'].upper().strip()
        try:
            stats_j = self._get_player_stats(jugador)
        except KeyError:
            stats_j = {col: DEFAULT_VALUES.get(col, 0) for col in COLS_JUGADORES[1:]}
            stats_j['jugador'] = jugador
        try:
            stats_e = self._get_team_stats(rival)
        except KeyError:
            stats_e = {col: 0 for col in COLS_EQUIPOS[1:]}
            stats_e['TEAM'] = rival
        try:
            stats_pos = self._get_position_stats(rival, jugador)
        except (KeyError, IndexError):
            stats_pos = {'ptspp': DEFAULT_VALUES['pts'], 
                         'rebpp': DEFAULT_VALUES['reb'], 
                         'asistpp': DEFAULT_VALUES['ast']}
        last_data = self._get_last_game_data(jugador, rival)

        features = {
            'min': input_data['mins'],
            'oponente_enc': self.feature_engineer.le.transform([rival])[0],
            'USG_perc': stats_j.get('USG_perc', DEFAULT_VALUES.get('pts', 0)),
            'TS_perc_x': stats_j.get('TS_perc', DEFAULT_VALUES.get('pts', 0)),
            'AST_perc_x': stats_j.get('AST_perc', DEFAULT_VALUES.get('ast', 0)),
            'REB_perc_x': stats_j.get('REB_perc', DEFAULT_VALUES.get('reb', 0)),
            'OffRtg': stats_e.get('OffRtg', 0),
            'DefRtg': stats_e.get('DefRtg', 0),
            'PACE': stats_e.get('PACE', 0),
            'AST_perc_y': stats_e.get('AST_perc', 0),
            'AST_TO': stats_e.get('AST_TO', 0),
            'REB_perc_y': stats_e.get('REB_perc', 0),
            'eFG_perc': stats_e.get('eFG_perc', 0),
            'PIE': stats_e.get('PIE', 0),
            'pts_last5_vs_opp': last_data.get('pts_last5_vs_opp', DEFAULT_VALUES['pts']),
            'reb_last5_vs_opp': last_data.get('reb_last5_vs_opp', DEFAULT_VALUES['reb']),
            'asist_last5_vs_opp': last_data.get('asist_last5_vs_opp', DEFAULT_VALUES['ast']),
            'FG_perc_last5_vs_opp': last_data.get('FG_perc_last5_vs_opp', DEFAULT_VALUES['FG_perc']),
            'pts_last10': last_data.get('pts_last10', DEFAULT_VALUES['pts']),
            'reb_last10': last_data.get('reb_last10', DEFAULT_VALUES['reb']),
            'asist_last10': last_data.get('asist_last10', DEFAULT_VALUES['ast']),
            'FG_perc_last10': last_data.get('FG_perc_last10', DEFAULT_VALUES['FG_perc']),
            'pts_last_season': last_data.get('pts_last_season', DEFAULT_VALUES['pts']),
            'reb_last_season': last_data.get('reb_last_season', DEFAULT_VALUES['reb']),
            'asist_last_season': last_data.get('asist_last_season', DEFAULT_VALUES['ast']),
            'FG_perc_last_season': last_data.get('FG_perc_last_season', DEFAULT_VALUES['FG_perc']),
            'ptspp': stats_pos.get('ptspp', DEFAULT_VALUES['pts']),
            'rebpp': stats_pos.get('rebpp', DEFAULT_VALUES['reb']),
            'asistpp': stats_pos.get('asistpp', DEFAULT_VALUES['ast']),
            'local': input_data['local'],
            'dias_descanso_jugador': self._get_last_rest_jugador(jugador),
            'dias_descanso_rival': self._get_last_rest_rival(rival),
            'ultimo_resultado_equipo': input_data['ult_resultado'],
            'partidos_temporada': last_data.get('partidos_temporada', 10),
            'partidos_totales': last_data.get('partidos_totales', 50)
        }

        return pd.DataFrame([features])

    def _make_predictions(self, features):
        return {
            stat: max(0, float(model.predict(features)[0]))
            for stat, model in self.models.items()
        }

    def _get_last_rest_jugador(self, jugador):
        jugador = jugador.upper()
        datos = self.df[self.df['jugador'] == jugador].sort_values('fecha')
        if len(datos) < 2:
            return 0
        return datos.iloc[-1]['dias_descanso_jugador']

    def _get_last_rest_rival(self, rival):
        rival = rival.upper()
        datos = self.df[self.df['oponente'] == rival].sort_values('fecha')
        if len(datos) < 2:
            return 0
        return datos.iloc[-1]['dias_descanso_rival']

    def _get_player_stats(self, jugador):
        player_data = self.feature_engineer.df_j[self.feature_engineer.df_j['jugador'] == jugador]
        if player_data.empty:
            raise KeyError(f"Jugador {jugador} no encontrado")
        return player_data.iloc[0]

    def _get_team_stats(self, team):
        team_data = self.feature_engineer.df_e[self.feature_engineer.df_e['TEAM'] == team]
        if team_data.empty:
            raise KeyError(f"Equipo {team} no encontrado")
        return team_data.iloc[0]

    def _get_position_stats(self, team, player):
        player_pos = self.df[self.df['jugador'] == player]['pos'].iloc[0].upper()
        team_pos_data = self.feature_engineer.df_p[
            (self.feature_engineer.df_p["TEAM"] == team) &
            (self.feature_engineer.df_p["pos"] == player_pos)
        ]
        if team_pos_data.empty:
            raise IndexError(f"No hay datos de posición {player_pos} para el equipo {team}")
        return team_pos_data.iloc[0]

    def _get_last_game_data(self, jugador, rival):
        jugador = jugador.upper()
        rival = rival.upper()
        player_data = self.df[self.df['jugador'] == jugador]
        if player_data.empty:
            return DEFAULT_VALUES
        last10 = player_data.tail(10)
        last5_vs_rival = player_data[player_data['oponente'] == rival].tail(5)
        current_season = player_data[player_data['temporada'] == '24/25']
        last_season = player_data[player_data['temporada'] == '23/24']
        result = {
            'pts_last5_vs_opp': last5_vs_rival['pts'].mean() if not last5_vs_rival.empty else None,
            'reb_last5_vs_opp': last5_vs_rival['reb'].mean() if not last5_vs_rival.empty else None,
            'asist_last5_vs_opp': last5_vs_rival['ast'].mean() if not last5_vs_rival.empty else None,
            'FG_perc_last5_vs_opp': last5_vs_rival['FG_perc'].mean() if not last5_vs_rival.empty else None,
            'pts_last10': last10['pts'].mean(),
            'reb_last10': last10['reb'].mean(),
            'asist_last10': last10['ast'].mean(),
            'FG_perc_last10': last10['FG_perc'].mean(),
            'pts_last_season': last_season['pts'].mean() if not last_season.empty else None,
            'reb_last_season': last_season['reb'].mean() if not last_season.empty else None,
            'asist_last_season': last_season['ast'].mean() if not last_season.empty else None,
            'FG_perc_last_season': last_season['FG_perc'].mean() if not last_season.empty else None,
            'partidos_temporada': len(current_season),
            'partidos_totales': len(player_data)
        }
        for stat in ['pts', 'reb', 'ast', 'FG_perc']:
            for window in ['last5_vs_opp', 'last10', 'last_season']:
                key = f"{stat}_{window}"
                if result.get(key) is None:
                    if window == 'last_season':
                        result[key] = result.get(f"{stat}_last10", DEFAULT_VALUES[stat])
                    else:
                        result[key] = DEFAULT_VALUES[stat]
        return result

    def _calculate_volatility(self, jugador):
        jugador = jugador.upper()
        player_data = self.df[self.df['jugador'] == jugador]
        if len(player_data) < 10:
            return {
                'pts': DEFAULT_VALUES['vol_pts'],
                'reb': DEFAULT_VALUES['vol_reb'],
                'ast': DEFAULT_VALUES['vol_ast']
            }
        current_season = player_data[player_data['temporada'] == '24/25']
        last_season = player_data[player_data['temporada'] == '23/24']

        def calc_vol(col, data):
            if len(data) < 5: return np.nan
            vals = data[col].astype(float)
            return vals.std() / (vals.mean() + 1e-6)

        results = {}
        for stat in ['pts', 'reb', 'ast']:
            vol_current = calc_vol(stat, current_season)
            vol_last = calc_vol(stat, last_season)
            vol_career = calc_vol(stat, player_data)

            total_games = len(player_data)
            available_weights = {
                'current': min(0.5, len(current_season) / 40),
                'last': min(0.3, len(last_season) / 60),
                'career': 0.2
            }
            total_weight = sum(available_weights.values())
            weights = {k: v / total_weight for k, v in available_weights.items()}

            vols, weights_list = [], []
            if not np.isnan(vol_current):
                vols.append(vol_current)
                weights_list.append(weights['current'])
            if not np.isnan(vol_last):
                vols.append(vol_last)
                weights_list.append(weights['last'])
            if not np.isnan(vol_career):
                vols.append(vol_career)
                weights_list.append(weights['career'])

            if len(vols) == 0:
                final_vol = DEFAULT_VALUES[f'vol_{stat}']
            else:
                final_vol = np.average(vols, weights=weights_list)
                final_vol *= min(1.2, 100 / total_games)

            results[stat] = np.clip(final_vol, 0.15, 0.45)
        return results

    def _handle_error(self, error):
        print(f"Error durante la predicción: {str(error)}")
# ==================== FUNCIONES AUXILIARES ====================

def cargar_log_apuestas():
    archivo_excel = "apuestas_log_simple.csv"
    if not os.path.isfile(archivo_excel):
        print(f"[INFO] No se encontró el archivo '{archivo_excel}'. No se usará histórico.")
        return pd.DataFrame()
    try:
        df_log = pd.read_csv(archivo_excel)
        for col in ['valor_esperado_EV', 'cuota_ofrecida', 'acierto']:
            if col in df_log.columns:
                df_log[col] = pd.to_numeric(df_log[col], errors='coerce')
        return df_log
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo histórico: {e}")
        return pd.DataFrame()

def obtener_metricas_historicas(df_log, stat, ev_min=0.0):
    if df_log.empty or 'stat' not in df_log.columns:
        return None
    df_filtrado = df_log[(df_log['stat'] == stat) & (df_log['valor_esperado_EV'] >= ev_min)]
    total = len(df_filtrado)
    if total == 0:
        return None
    aciertos = df_filtrado['acierto'].sum()
    ev_medio = df_filtrado['valor_esperado_EV'].mean()
    beneficio = ((df_filtrado['acierto'] * (df_filtrado['cuota_ofrecida'] - 1)) - (1 - df_filtrado['acierto'])).sum()
    return {
        'total': total,
        'acierto_pct': aciertos / total,
        'ev_medio': ev_medio,
        'beneficio': beneficio
    }

def recomendar_seguridad(prob, vol):
    if prob > 0.62 and vol < 0.28:
        return "Alta"
    elif prob > 0.57 and vol < 0.34:
        return "Media"
    elif prob > 0.52 and vol < 0.40:
        return "Moderada"
    else:
        return "Baja"

def guardar_apuesta(jugador, rival, stat, linea, cuota_ofrecida, cuota_justa, prob, vol, recomendacion, seguridad, pred, percentiles, ev):
    archivo_csv = "apuestas_log_simple.csv"
    campos = [
        "fecha", "jugador", "rival", "stat", "linea_apuesta", "cuota_ofrecida", "cuota_justa",
        "probabilidad", "valor_esperado_EV", "recomendacion", "seguridad", "acierto", "proba_XGB"
    ]
    fila = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        jugador,
        rival,
        stat,
        f"{linea:.2f}",
        f"{cuota_ofrecida:.2f}",
        f"{cuota_justa:.2f}" if cuota_justa is not None else "",
        f"{prob:.2f}",
        f"{ev:.3f}" if ev is not None else "",
        recomendacion,
        seguridad,
        0,      # acierto a rellenar manualmente
        ""      # proba_XGB SIEMPRE vacío
    ]
    try:
        write_header = not os.path.isfile(archivo_csv) or os.path.getsize(archivo_csv) == 0
        with open(archivo_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(campos)
            writer.writerow(fila)
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo guardar la apuesta: {e}")

def advertencia_partidos(df, jugador):
    n = df[df['jugador'] == jugador.upper()].shape[0]
    if n < 10:
        print(f"\033[93m[ADVERTENCIA] Solo hay {n} partidos para {jugador}. Predicción y volatilidad poco fiables.\033[0m")
    elif n < 20:
        print(f"\033[93m[INFO] Hay {n} partidos para {jugador}. Mejorable pero aceptable.\033[0m")

def display_results_con_historial(preds, vol, results, lines, df, jugador, rival, df_log):
    advertencia_partidos(df, jugador)
    print("\nRESULTADOS PREDICHOS:")
    print(f"Puntos: {preds['pts']:.1f} | Rebotes: {preds['reb']:.1f} | Asistencias: {preds['ast']:.1f}")
    print("\nVOLATILIDAD (0.4+ = alta):")
    print(f"Puntos: {vol['pts']:.2f} | Rebotes: {vol['reb']:.2f} | Asistencias: {vol['ast']:.2f}")
    print("\nDETALLES DE SIMULACIÓN (LOG-NORMAL):")
    for stat in ['pts', 'reb', 'ast']:
        res = results[stat]
        cuota_justa = 1 / res['probability'] if res['probability'] > 0 else None
        cuota_ofrecida = lines[f"{stat}_cuota"]
        ev = res['probability'] * cuota_ofrecida - 1 if cuota_ofrecida and res['probability'] > 0 else None
        seguridad = recomendar_seguridad(res['probability'], vol[stat])

        met_historicas = obtener_metricas_historicas(df_log, stat, ev_min=0.0)

        print(f"\n • {stat.upper()}:")
        print(f"   - Línea: {lines[stat]}")
        print(f"   - Probabilidad Over: {res['probability']*100:.1f}%")
        print(f"   - Cuota justa (modelo): {cuota_justa:.2f}" if cuota_justa else "   - Cuota justa: N/A")
        print(f"   - Cuota ofrecida: {cuota_ofrecida:.2f}")
        print(f"   - Valor Esperado (EV): {ev:.3f}" if ev is not None else "   - Valor Esperado (EV): N/A")
        print(f"   - Seguridad: {seguridad}")

        if met_historicas is None:
            print("   - Sin datos históricos suficientes para evaluar recomendaciones.")
            recomendacion = "Usar criterio EV actual"
        else:
            print(f"   - Histórico apuestas {stat}: {met_historicas['total']} apuestas")
            print(f"     % acierto histórico: {met_historicas['acierto_pct']*100:.1f}%")
            print(f"     EV medio histórico: {met_historicas['ev_medio']:.3f}")
            print(f"     Beneficio acumulado histórico: {met_historicas['beneficio']:.2f}")

            if ev is not None and ev > 0 and met_historicas['acierto_pct'] > 0.5 and met_historicas['beneficio'] > 0:
                print("\033[92m   >> Recomendación: APOSTAR (EV positivo + buen histórico)\033[0m")
                recomendacion = "APOSTAR (EV positivo + buen histórico)"
            else:
                print("\033[91m   >> Recomendación: NO apostar (histórico no favorable o EV negativo)\033[0m")
                recomendacion = "NO apostar"

        print(f"   - Percentiles:")
        print(f"     5%: {res['percentiles'][0]:.1f} | 25%: {res['percentiles'][1]:.1f} | 50%: {res['percentiles'][2]:.1f}")
        print(f"     75%: {res['percentiles'][3]:.1f} | 95%: {res['percentiles'][4]:.1f}")
        print(f"   - Simulaciones realizadas: {res['simulations']}")

        guardar_apuesta(
            jugador, rival, stat, lines[stat], cuota_ofrecida, cuota_justa, res['probability'],
            vol[stat], recomendacion, seguridad, preds[stat], res['percentiles'], ev
        )

def get_user_input():
    print("\nPREDICCIÓN NBA - Temporadas 23.24 y 24.25")
    print("----------------------------------------")
    jugador = input("\nJugador (ej. LeBron James): ").strip()
    rival = input("Rival (ej. LAL): ").strip()
    mins = float(input("Minutos estimados: "))
    local_input = input("¿El partido es local? (s/n): ").strip().lower()
    local = 1 if local_input == 's' else 0
    ult_resultado_input = input("Último resultado del equipo (v = victoria, d = derrota): ").strip().lower()
    ult_resultado = 1 if ult_resultado_input == 'v' else 0
    return {
        'jugador': jugador,
        'rival': rival,
        'mins': mins,
        'local': local,
        'ult_resultado': ult_resultado
    }

def get_lines_input():
    print("\nIntroduce las líneas y cuotas para las apuestas:")
    return {
        'pts': float(input("Over Puntos línea (ej. 25.5): ")),
        'pts_cuota': float(input("Cuota Over Puntos ofrecida (ej. 1.85): ")),
        'reb': float(input("Over Rebotes línea (ej. 7.5): ")),
        'reb_cuota': float(input("Cuota Over Rebotes ofrecida (ej. 1.85): ")),
        'ast': float(input("Over Asistencias línea (ej. 8.5): ")),
        'ast_cuota': float(input("Cuota Over Asistencias ofrecida (ej. 1.85): "))
    }

def analizar_log_apuestas():
    archivo_excel = "apuestas_log_simple.csv"
    if not os.path.isfile(archivo_excel):
        print(f"No se encontró el archivo '{archivo_excel}'.")
        return
    try:
        df = pd.read_csv(archivo_excel)
    except Exception as e:
        print(f"Error al leer '{archivo_excel}': {e}")
        return
    if len(df) < 10:
        print(f"Solo hay {len(df)} apuestas registradas. Se necesitan al menos 10 para análisis fiable.")
        return
    print("\n--- ANÁLISIS DE APUESTAS GUARDADAS ---")
    print(f"Apuestas registradas: {len(df)}")
    if 'acierto' in df.columns:
        porcentaje_acierto = df['acierto'].mean() * 100
        print(f"Porcentaje de aciertos global: {porcentaje_acierto:.1f}%")
    else:
        print("No se encontró columna 'acierto'.")
    if 'valor_esperado_EV' in df.columns:
        ev_medio = df['valor_esperado_EV'].mean()
        print(f"EV medio de las apuestas: {ev_medio:.3f}")
    else:
        print("No se encontró columna 'valor_esperado_EV'.")
    if 'cuota_ofrecida' in df.columns and 'acierto' in df.columns:
        beneficio_total = ((df['acierto'] * (df['cuota_ofrecida'] - 1)) - (1 - df['acierto'])).sum()
        print(f"Beneficio acumulado (unidades): {beneficio_total:.2f}")
    else:
        print("No se encontró columnas necesarias para calcular beneficio acumulado.")
    if 'stat' in df.columns and 'acierto' in df.columns:
        acierto_stat = df.groupby('stat')['acierto'].mean() * 100
        print("\nAcierto por tipo de stat:")
        print(acierto_stat.to_string())
    else:
        print("No se encontró columnas necesarias para acierto por tipo de stat.")
    if 'stat' in df.columns and 'valor_esperado_EV' in df.columns:
        ev_stat = df.groupby('stat')['valor_esperado_EV'].mean()
        print("\nEV medio por tipo de stat:")
        print(ev_stat.to_string())
    else:
        print("No se encontró columnas necesarias para EV medio por tipo de stat.")
# ==================== MAIN ====================

def main():
    print("Elige una opción:")
    print("1 - Realizar predicción (usa los modelos guardados o entrena por defecto)")
    print("2 - Analizar apuestas guardadas")
    print("3 - Entrenar modelos XGBoost con hiperparámetros (GridSearchCV)")

    opcion = input("Opción (1/2/3): ").strip()

    if opcion == '3':
        data_loader = NBADataLoader()
        df, df_j, df_e, df_p = data_loader.load_all_data()
        feature_engineer = FeatureEngineer(df, df_j, df_e, df_p)
        X = feature_engineer.construir_features()
        y_dict = feature_engineer.get_targets()
        model_trainer = NBAModelTrainer(X, y_dict)
        model_trainer.train_with_gridsearch()
        print("\n¡Puedes ahora usar la opción 1 para predecir con estos modelos nuevos!")
        return

    elif opcion == '1':
        try:
            # 1. Cargar datos y features
            data_loader = NBADataLoader()
            df, df_j, df_e, df_p = data_loader.load_all_data()
            feature_engineer = FeatureEngineer(df, df_j, df_e, df_p)
            X = feature_engineer.construir_features()
            y_dict = feature_engineer.get_targets()

            # 2. Cargar modelos ya entrenados, o entrenar por defecto si no existen
            models = {}
            for stat in ['pts', 'reb', 'ast']:
                model_path = f'models/model_{stat}.json'
                if os.path.exists(model_path):
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    models[stat] = model
                else:
                    print(f"Modelo para {stat} no encontrado, entrenando uno básico...")
                    model_trainer = NBAModelTrainer(X, y_dict)
                    models = model_trainer.train_with_gridsearch()
                    break  # Entrena los 3 juntos y sale del bucle

            # 3. Pedir datos al usuario y predecir
            input_data = get_user_input()
            predictor = NBAPredictor(models, feature_engineer)
            preds, vol = predictor.predict(input_data)

            # 4. Monte Carlo + mostrar resultados
            montecarlo = MonteCarloSimulator(random_state=42)
            lines = get_lines_input()
            results = montecarlo.calculate_probabilities(preds, vol, {
                'pts': lines['pts'],
                'reb': lines['reb'],
                'ast': lines['ast']
            })

            df_log = cargar_log_apuestas()

            display_results_con_historial(
                preds, vol, results, lines,
                df, input_data['jugador'], input_data['rival'], df_log
            )

        except Exception as e:
            logging.error(f"Error en la ejecución: {str(e)}", exc_info=True)
            print(f"\nError: {str(e)}")
            print("Por favor, verifica los datos e intenta nuevamente.")

    elif opcion == '2':
        analizar_log_apuestas()
    else:
        print("Opción no válida. Por favor elige 1, 2 o 3.")

if __name__ == "__main__":
    main()
