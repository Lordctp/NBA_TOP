import pandas as pd

# 1. Cargar el archivo alineaciones.csv
df = pd.read_csv('alineaciones.csv')

# 2. Para cada GAME_ID y TEAM, quedarte solo con las primeras 5 filas (el resto se elimina)
df_filtrado = df.groupby(['GAME_ID', 'TEAM'], as_index=False).head(5)

# 3. Sobrescribir el archivo original
df_filtrado.to_csv('alineaciones.csv', index=False)

print("¡Listo! El archivo 'alineaciones.csv' ahora solo tiene 5 jugadores por equipo y partido.")

