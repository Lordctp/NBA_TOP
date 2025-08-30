import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

LOG_FILE = "apuestas_log_simple.csv"
N_MINIMO = 100

def preparar_features_apuestas(df):
    features_df = pd.DataFrame(index=df.index)
    features_df["cuota_ofrecida"] = df["cuota_ofrecida"]
    features_df["cuota_round"] = df["cuota_ofrecida"].round(2)
    features_df["probabilidad"] = df["probabilidad"]
    features_df["valor_esperado_EV"] = df["valor_esperado_EV"]
    features_df["acierto"] = df["acierto"]
    features_df["stat"] = df["stat"] if "stat" in df.columns else ""
    features_df["jugador"] = df["jugador"] if "jugador" in df.columns else ""
    features_df["rival"] = df["rival"] if "rival" in df.columns else ""

    cuota_media = features_df.groupby("cuota_round")["acierto"].transform("mean")
    features_df["acierto_cuota"] = cuota_media
    features_df["hit_jugador_last10"] = features_df.groupby(["jugador", "stat"])["acierto"].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    features_df["hit_jugador_last5"]  = features_df.groupby(["jugador", "stat"])["acierto"].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    features_df["hit_jugador_vs_rival"] = features_df.groupby(["jugador", "rival", "stat"])["acierto"].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    features_df["hit_global_stat"] = features_df.groupby("stat")["acierto"].transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
    
    # Nueva feature: proba_XGB_prev (shifted)
    if "proba_XGB" in df.columns:
        features_df["proba_XGB_prev"] = df.groupby(["jugador", "stat"])["proba_XGB"].shift(1)
    else:
        features_df["proba_XGB_prev"] = -1

    features_df = features_df.fillna({
        "hit_jugador_last10": -1,
        "hit_jugador_last5": -1,
        "hit_jugador_vs_rival": -1,
        "hit_global_stat": -1,
        "proba_XGB_prev": -1,
        "acierto_cuota": -1
    })

    features_final = [
        "cuota_ofrecida", "cuota_round", "probabilidad", "valor_esperado_EV", 
        "hit_jugador_last10", "hit_jugador_last5", "hit_jugador_vs_rival", "hit_global_stat",
        "acierto_cuota", "proba_XGB_prev"
    ]
    return features_df[features_final]

def main():
    df = pd.read_csv(LOG_FILE)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors='coerce')

    if len(df) < N_MINIMO:
        df["proba_XGB"] = ""
    else:
        # Si existe ya proba_XGB del pasado, la reutilizamos para el feature proba_XGB_prev
        if "proba_XGB" not in df.columns:
            df["proba_XGB"] = np.nan  # para poder crear proba_XGB_prev

        features_df = preparar_features_apuestas(df)
        target = df["acierto"]
        valid_rows = features_df.loc[N_MINIMO:].dropna()
        y_valid = target.loc[valid_rows.index]
        if len(valid_rows) >= 50:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.12,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
            model.fit(valid_rows, y_valid)
            pred_mask = df.index >= N_MINIMO
            X_pred = features_df.loc[pred_mask].fillna(-1)
            proba_xgb = model.predict_proba(X_pred)[:,1]
            df["proba_XGB"] = 0
            df.loc[pred_mask, "proba_XGB"] = np.round(proba_xgb, 3)
            df.loc[:N_MINIMO-1, "proba_XGB"] = 0
        else:
            df["proba_XGB"] = 0
            df.loc[:N_MINIMO-1, "proba_XGB"] = 0

    df.to_csv(LOG_FILE, index=False)

    # Mostrar SOLO la columna proba_XGB de las apuestas del día
    if "fecha" in df.columns:
        hoy = datetime.now().date()
        mask_hoy = df["fecha"].dt.date == hoy
        print(df.loc[mask_hoy, ["proba_XGB"]])
    else:
        print("No se puede mostrar proba_XGB del día porque no existe columna 'fecha'.")

if __name__ == "__main__":
    main()
