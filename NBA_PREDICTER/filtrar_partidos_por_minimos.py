# -*- coding: utf-8 -*-
"""
eliminar_jugadores_lista_2.py
-----------------------------
Elimina del archivo 'partidos.csv' todas las filas correspondientes a la
siguiente lista de jugadores y SOBRESCRIBE el archivo.
Por defecto crea una copia de seguridad automática, salvo que se use --no-backup.

Jugadores objetivo (del mensaje más reciente):
    - Jaylon Tyson
    - Terrence Shannon Jr.
    - Bones Hyland
    - Anthony Gill
    - Trevelin Queen
    - Rob Dillingham
    - Charles Bassey
    - Olivier-Maxence Prosper
    - Matt Ryan
    - Craig Porter Jr.
    - Jaden Springer
    - Marjon Beauchamp
    - Quenton Jackson
    - Moses Brown
    - Sandro Mamukelashvili
"""

import argparse
from pathlib import Path
from datetime import datetime
import re
import pandas as pd


JUGADORES_A_ELIMINAR = [
    "Jaylon Tyson",
    "Terrence Shannon Jr.",
    "Bones Hyland",
    "Anthony Gill",
    "Trevelin Queen",
    "Rob Dillingham",
    "Charles Bassey",
    "Olivier-Maxence Prosper",
    "Matt Ryan",
    "Craig Porter Jr.",
    "Jaden Springer",
    "Marjon Beauchamp",
    "Quenton Jackson",
    "Moses Brown",
    "Sandro Mamukelashvili",
]


def normalize_name(s: str) -> str:
    """Normaliza nombre: mayúsculas, quita puntuación simple y colapsa espacios.
    Quitamos . , ' - para cubrir variantes como 'Jr.' / 'Jr' o nombres con guión.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.upper().strip()
    s = re.sub(r"[.,'\-]", "", s)    # quitar . , ' -
    s = re.sub(r"\s+", " ", s)       # colapsar espacios
    return s


def detect_column(df, candidates):
    """Devuelve la primera columna de 'candidates' que exista."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Elimina jugadores listados de partidos.csv (in-place).")
    parser.add_argument("--csv", default="partidos.csv", help="Ruta del CSV (por defecto partidos.csv).")
    parser.add_argument("--no-backup", action="store_true",
                        help="No crear copia de seguridad del archivo original (no recomendado).")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[ERROR] No se encuentra '{path}'.")
        raise SystemExit(1)

    # Leer CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] No se pudo leer '{path}': {e}")
        raise SystemExit(1)

    # Detectar columna de jugador
    col_player = detect_column(df, ["jugador", "Player", "player", "JUGADOR"])
    if col_player is None:
        print("[ERROR] No se encontró columna de jugador ('jugador' o 'Player').")
        print(f"Columnas disponibles: {list(df.columns)}")
        raise SystemExit(1)

    # Normalizaciones
    df["_player_norm"] = df[col_player].apply(normalize_name)
    to_remove_norm = {normalize_name(p) for p in JUGADORES_A_ELIMINAR}

    # Resumen previo
    total_before = len(df)
    mask_remove = df["_player_norm"].isin(to_remove_norm)
    remove_counts = df.loc[mask_remove, "_player_norm"].value_counts().to_dict()
    n_remove = int(mask_remove.sum())

    print("---- RESUMEN PREVIO ----")
    print(f"Archivo:        {path.name}")
    print(f"Filas totales:  {total_before}")
    print(f"Filas a borrar: {n_remove}")
    if remove_counts:
        print("\nEliminado por jugador (tras normalización):")
        for name, cnt in sorted(remove_counts.items()):
            print(f" - {name.title():<25}  filas={cnt}")

    # Filtrar y quitar columna auxiliar
    df_filtered = df.loc[~mask_remove].drop(columns=["_player_norm"])
    total_after = len(df_filtered)

    # Backup opcional y escritura in-place
    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        try:
            path.rename(backup)
            print(f"[OK] Copia de seguridad: {backup.name}")
        except Exception as e:
            print(f"[ERROR] No se pudo crear backup de '{path}': {e}")
            print("Por seguridad, se cancelan cambios.")
            raise SystemExit(1)
        try:
            df_filtered.to_csv(path, index=False)
            print(f"[OK] Archivo sobrescrito: {path.name}")
        except Exception as e:
            print(f"[ERROR] No se pudo escribir '{path}': {e}")
            print("Restaurando backup...")
            try:
                if path.exists():
                    path.unlink()
                backup.rename(path)
                print("[OK] Backup restaurado.")
            except Exception as e2:
                print(f"[ERROR] No se pudo restaurar backup: {e2}")
            raise SystemExit(1)
    else:
        # Sin backup
        try:
            df_filtered.to_csv(path, index=False)
            print(f"[OK] Archivo sobrescrito SIN backup: {path.name}")
        except Exception as e:
            print(f"[ERROR] No se pudo escribir '{path}': {e}")
            raise SystemExit(1)

    print("\n---- RESULTADO ----")
    print(f"Eliminadas:  {n_remove}")
    print(f"Restantes:   {total_after}")


if __name__ == "__main__":
    main()
