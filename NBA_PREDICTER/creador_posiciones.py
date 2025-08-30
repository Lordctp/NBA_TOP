#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera medias por equipo rival y posición a partir de partidos.csv
y guarda posiciones_jugador.txt con formato estilo CSV.

- Filtra jugadores con >= 28 minutos.
- Selecciona temporada: usa 25/26 si existe; si no 24/25; si no la mayor disponible.
- Ignora fecha y localía.
- Calcula medias de todas las stats y FG%/3PT% con totales.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Archivos
CSV_IN = Path("partidos.csv")
CSV_OUT = Path("posiciones_jugador.txt")

def main():
    if not CSV_IN.exists():
        sys.exit(f"No se encontró el archivo: {CSV_IN.resolve()}")

    df = pd.read_csv(CSV_IN, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Player","Opp","pts","reb","ast","min","stl","blk","fga","fgm","fgp",
        "3pa","3pm","3pp","tov","pf","fta","pos","temp","fecha","local"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"Faltan columnas: {missing}")

    num_cols = ["pts","reb","ast","min","stl","blk","fga","fgm","fgp","3pa","3pm","3pp","tov","pf","fta"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filtramos minutos
    df = df[df["min"] >= 28]
    if df.empty:
        sys.exit("Sin datos con >=28 minutos.")

    # Temporada a usar
    seasons = df["temp"].astype(str).str.strip().unique().tolist()
    if "25/26" in seasons:
        season_to_use = "25/26"
    elif "24/25" in seasons:
        season_to_use = "24/25"
    else:
        season_to_use = max(seasons)

    df = df[df["temp"].astype(str).str.strip() == season_to_use]

    # Agrupación
    grouped = df.groupby(["Opp","pos"], dropna=True)
    mean_cols = ["pts","reb","ast","min","stl","blk","fga","fgm","3pa","3pm","tov","pf","fta"]
    means = grouped[mean_cols].mean()

    totals = grouped[["fgm","fga","3pm","3pa"]].sum(min_count=1).reset_index()
    totals["fgp"] = np.where(totals["fga"] > 0, totals["fgm"] / totals["fga"], np.nan)
    totals["3pp"] = np.where(totals["3pa"] > 0, totals["3pm"] / totals["3pa"], np.nan)
    pcts = totals.set_index(["Opp","pos"])[["fgp","3pp"]]

    counts = grouped.size().to_frame("n_player_games")

    result = means.join(pcts, how="left").join(counts, how="left")

    # Reordenamos columnas al estilo partidos.csv
    result = result[
        ["pts","reb","ast","min","stl","blk",
         "fga","fgm","fgp","3pa","3pm","3pp",
         "tov","pf","fta","n_player_games"]
    ].round(3)

    # Guardar con cabecera estilo CSV (sep=",")
    result.reset_index().to_csv(CSV_OUT, index=False, encoding="utf-8")

    print(f"✔ Archivo generado: {CSV_OUT.resolve()}")
    print(f"✔ Temporada usada: {season_to_use}")

if __name__ == "__main__":
    main()



