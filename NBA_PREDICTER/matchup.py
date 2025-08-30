import pandas as pd

JUGADORES = [
   "Mikal Bridges", "Josh Hart", "Anthony Edwards", "Devin Booker", "James Harden",
"DeMar DeRozan", "Trae Young", "Tyler Herro", "OG Anunoby", "Jalen Green",
"Christian Braun", "Bam Adebayo", "Ivica Zubac", "Jayson Tatum", "Jaden McDaniels",
"Keegan Murray", "Zach LaVine", "Shai Gilgeous-Alexander", "Michael Porter Jr.", "Derrick White",
"Dyson Daniels", "Nikola Jokic", "Austin Reaves", "Toumani Camara", "Pascal Siakam",
"Brook Lopez", "Karl-Anthony Towns", "Bub Carrington", "Cade Cunningham", "Tyrese Haliburton",
"Coby White", "LeBron James", "Domantas Sabonis", "Jamal Murray", "Alperen Sengun",
"Dillon Brooks", "Rudy Gobert", "Tobias Harris", "Darius Garland", "Jalen Brunson",
"Jarrett Allen", "Chris Paul", "Anfernee Simons", "Giannis Antetokounmpo", "Malik Beasley",
"Kentavious Caldwell-Pope", "Nikola Vucevic", "Payton Pritchard", "Kevin Durant", "Shaedon Sharpe",
"Stephen Curry", "De'Aaron Fox", "Jalen Williams", "Donovan Mitchell", "Harrison Barnes",
"Julius Randle", "Amen Thompson", "Jaren Jackson Jr.", "Desmond Bane", "Naz Reid",
"Tyus Jones", "Myles Turner", "Evan Mobley", "Taurean Prince", "Stephon Castle",
"Deni Avdija", "Jaylen Brown", "Tim Hardaway Jr.", "Bennedict Mathurin", "Spencer Dinwiddie",
"Scottie Barnes", "Max Christie", "Josh Giddey", "Fred VanVleet", "Keyonte George",
"Dennis Schroder", "Damian Lillard", "Russell Westbrook", "Kelly Oubre Jr.", "Nickeil Alexander-Walker",
"Luguentz Dort", "Onyeka Okongwu", "Malik Monk", "Jaylen Wells", "Jalen Duren",
"Jalen Wilson", "Miles Bridges", "Davion Mitchell", "Franz Wagner", "Quentin Grimes",
"Jordan Poole", "Draymond Green", "Devin Vassell", "Klay Thompson", "Tyrese Maxey",
"Norman Powell", "Yves Missi", "Keon Ellis", "Bilal Coulibaly", "Kyle Kuzma",
"Julian Champagnie", "Keon Johnson", "Naji Marshall", "Jrue Holiday", "Guerschon Yabusele",
"Gary Trent Jr.", "Anthony Black", "Josh Green", "Nic Claxton", "Andrew Nembhard",
"Cason Wallace", "Derrick Jones Jr.", "Rui Hachimura", "RJ Barrett", "Buddy Hield",
"Trey Murphy III", "Zaccharie Risacher", "Andrew Wiggins", "Isaiah Collier", "Keldon Johnson",
"Royce O'Neale", "P.J. Washington", "CJ McCollum", "Haywood Highsmith", "Dorian Finney-Smith",
"Alex Sarr", "Kyrie Irving", "Kyshawn George", "Cameron Johnson", "Duncan Robinson",
"Kris Dunn", "Luka Doncic", "Scoot Henderson", "Wendell Carter Jr.", "Collin Sexton",
"Mike Conley", "Amir Coffey", "Jimmy Butler", "Aaron Wiggins", "Walker Kessler",
"De'Andre Hunter", "Ochai Agbaji", "Brandin Podziemski", "Jabari Smith Jr.", "Anthony Davis",
"Bradley Beal", "Georges Niang", "Jakob Poeltl", "Scotty Pippen Jr.", "Kevin Huerter",
"Santi Aldama", "Al Horford", "A.J. Green", "Terry Rozier", "Peyton Watson",
"Moses Moody", "Tristan Da Silva", "Donte DiVincenzo", "Isaiah Joe", "Corey Kispert",
"Caris LeVert", "Miles McBride", "Isaiah Hartenstein", "Gradey Dick", "Paolo Banchero",
"Patrick Williams", "Obi Toppin", "Grayson Allen", "Sam Hauser", "Ziaire Williams",
"Victor Wembanyama", "Gabe Vincent", "Jonas Valanciunas", "Jerami Grant", "Kyle Filipowski",
"Ja Morant", "Matas Buzelis", "LaMelo Ball", "Dalton Knecht", "Kevin Porter Jr.",
"D'Angelo Russell", "Lauri Markkanen", "Luke Kennard", "Jamal Shead", "Aaron Gordon",
"Isaiah Stewart", "Brice Sensabaugh", "Goga Bitadze", "Kel'el Ware", "Tari Eason",
"Zach Edey", "T.J. McConnell", "Terance Mann", "Ryan Dunn", "Sam Merrill",
"Ayo Dosunmu", "Ty Jerome", "Julian Strawther", "Nick Smith Jr.", "Jaime Jaquez Jr.",
"Nicolas Batum", "Jose Alvarado", "Jeremy Sochan", "Luke Kornet", "Jake LaRavia",
"Trey Lyles", "Bogdan Bogdanovic", "Justin Champagnie", "Paul George", "Ausar Thompson",
"Donovan Clingan", "Jordan Hawkins", "Tyrese Martin", "Mason Plumlee", "Jonathan Mogbo",
"Jalen Johnson", "Max Strus", "Johnny Juzang", "Ron Holland", "Javonte Green",
"Nick Richards", "Dean Wade", "Ricky Council IV", "Bobby Portis", "Tidjane Salaun",
"Moussa Diabate", "Jeremiah Robinson-Earl", "Simone Fontecchio", "Cole Anthony", "Ben Sheppard",
"Daniel Gafford", "John Collins", "Caleb Martin", "Kristaps Porzingis", "Brandon Clarke",
"Deandre Ayton", "Jarace Walker", "Kawhi Leonard", "Clint Capela", "Cody Martin",
"Mark Williams", "Precious Achiuwa", "Nikola Jovic", "Justin Edwards", "Vit Krejci",
"Jonathan Kuminga", "Kevon Looney", "Kenrich Williams", "Aaron Nesmith", "Julian Phillips",
"Dalano Banton", "Ben Simmons", "Ja'Kobe Walter", "Jaxson Hayes", "Jonathan Isaac",
"Cameron Payne", "DaQuan Jeffries", "Jusuf Nurkic", "Seth Curry", "Cody Williams",
"Isaac Okoro", "Oso Ighodaro", "Jared Butler", "Jamison Battle", "Noah Clowney",
"Alex Caruso", "Kris Murray", "Dejounte Murray", "Kyle Anderson", "Jalen Suggs",
"Brandon Boston Jr.", "Dalen Terry", "Andre Jackson Jr.", "Zach Collins", "Trayce Jackson-Davis",
"Thomas Bryant", "Jalen Smith", "Jordan Clarkson", "Gary Payton II", "Brandon Miller",
"Bruce Brown", "Immanuel Quickley", "Trendon Watford", "KJ Martin", "Day'Ron Sharpe",
"Jaden Hardy", "Adem Bona", "Tre Jones", "Jaden Ivey", "Kelly Olynyk",
"Chet Holmgren", "Neemias Queta", "Alec Burks", "Chris Boucher", "Zion Williamson",
"Khris Middleton", "KJ Simpson", "Shake Milton", "Dereck Lively II", "Garrison Mathews",
"Cam Whitmore", "Ryan Rollins", "Marcus Sasser", "Steven Adams", "Aaron Holiday",
"Karlo Matkovic", "Caleb Houstan", "Vasilije Micic", "Jaylin Williams", "Pelle Larsson",
"Cam Thomas", "Lindy Waters III", "Lonzo Ball", "Orlando Robinson", "Eric Gordon",
"Gui Santos", "Landry Shamet", "Svi Mykhailiuk", "Andre Drummond", "Jabari Walker",
"Jay Huff", "Drew Eubanks", "Talen Horton-Tucker", "Gary Harris", "Micah Potter",
"Jett Howard", "DeAndre Jordan", "Quinten Post", "Blake Wesley", "Sandro Mamukelashvili",
"Marcus Smart", "Jalen Pickett", "Tosan Evbuomwan", "Antonio Reeves", "Kyle Lowry",
"Reed Sheppard", "Herbert Jones", "AJ Johnson", "Maxi Kleber", "Delon Wright",
"Jericho Sims", "Josh Okogie", "Jeff Dowtin Jr.", "Daniel Theis", "Cory Joseph",
"Zeke Nnaji", "Kessler Edwards", "Pat Connaughton", "Ajay Mitchell", "Brandon Ingram",
"Jared McCain", "Cam Reddish", "Jae'Sean Tate", "Olivier-Maxence Prosper", "Jarred Vanderbilt",
"Joel Embiid", "Monte Morris", "Moritz Wagner", "Malcolm Brogdon", "John Konchar",
"Dwight Powell", "Dillon Jones", "Jordan Goodwin", "Richaun Holmes", "Mouhamed Gueye",
"Bismack Biyombo", "Jaylen Clark", "Rob Dillingham", "Colby Jones", "Tristan Vukcevic",
"Craig Porter Jr.", "Elfrid Payton", "Jamal Cain", "Keaton Wallace", "Jock Landale",
"Wendell Moore Jr.", "Vince Williams Jr.", "Brandon Williams", "A.J. Lawson", "Grant Williams",
"Lonnie Walker IV", "Duop Reath", "Reece Beekman", "Kai Jones", "Larry Nance Jr.",
"Collin Gillespie", "Rayan Rupert", "GG Jackson II", "Jaylon Tyson", "Bol Bol",
"Paul Reed", "Colin Castleton", "Trevelin Queen", "Malaki Branham", "Jordan Miller",
"Mo Bamba", "David Roddy", "Taj Gibson", "Ousmane Dieng", "Jordan Walsh",
"Hunter Tyson", "Anthony Gill", "Jeff Green", "Devin Carter", "Trey Jemison",
"Baylor Scheierman", "Reggie Jackson", "Quenton Jackson", "Alex Len", "Johnny Furphy",
"Charles Bassey", "Dominick Barlow", "Dante Exum", "Damion Baugh", "Jaden Springer",
"Robert Williams", "Christian Koloko", "Doug McDermott", "Terrence Shannon Jr.", "Keion Brooks Jr.",
"Tristan Thompson", "Maxwell Lewis", "Jevon Carter", "Patty Mills", "Tre Mann",
"Jalen Hood-Schifino", "Torrey Craig", "Jordan McLaughlin", "Matisse Thybulle", "Isaac Jones",
"Tyler Kolek", "Mitchell Robinson", "Josh Minott", "Kobe Brown", "Dru Smith",
"Marvin Bagley III", "Isaiah Wong", "Oscar Tshiebwe", "Drew Timme", "Marcus Bagley",
"Cam Spencer", "Kevin Love", "Pat Spencer", "JT Thor", "Branden Carlson",
"Dariq Whitehead", "Johnny Davis", "Bones Hyland", "Jaylen Martin", "Xavier Tillman Sr.",
"Jared Rhoden", "Garrett Temple", "Luka Garza", "Ariel Hukporti", "Alex Reese",
"Dario Saric", "Adam Flagler", "Chuma Okeke", "Drew Peterson", "Markelle Fultz",
"Lester Quinones", "Enrique Freeman", "Bronny James", "Cole Swider", "Jaylen Nowell",
"Markieff Morris", "Killian Hayes", "MarJon Beauchamp", "Lamar Stevens", "Tyson Etienne",
"Josh Richardson", "Bruno Fernando", "Jeenathan Williams", "Damion Lee", "Oshae Brissett",
"Pete Nance", "Elijah Harkless", "Vlatko Cancar", "Kylor Kelley", "Alex Ducas",
"Spencer Jones", "Kobe Bufkin", "Tyler Smith", "De'Anthony Melton", "Moses Brown",
"TyTy Washington Jr.", "Trey Alexander", "Sidy Cissoko", "Joe Ingles", "Tony Bradley",
"Jaylen Sims", "Pacome Dadiet", "Patrick Baldwin Jr.", "Chris Livingston", "Jae Crowder",
"Justin Minaya", "Keshad Johnson", "Yuki Kawamura", "JD Davison", "Luke Travers",
"Stanley Umude", "Isaiah Jackson", "Kevin Knox", "Marcus Garrett", "Jacob Toppin",
"Emoni Bates", "Taylor Hendricks", "Chris Duarte", "Daeqwon Plowden", "RayJ Dennis",
"Josh Christopher", "Matt Ryan", "Armel Traore", "PJ Hall", "Nae'Qwan Tomlin",
"Jamaree Bouyea", "Cam Christie", "P.J. Tucker", "Jazian Gortman", "Liam Robbins",
"E.J. Liddell", "Jackson Rowe", "N'Faly Dante", "Isaiah Crawford", "Mason Jones",
"Malachi Flynn", "Phillip Wheeler", "Jack McVeigh", "Bobi Klintman", "James Johnson",
"Harrison Ingram", "PJ Dozier", "Miles Norris", "David Duke Jr.", "D.J. Carton",
"Leonard Miller", "Ulrich Chomche", "Bryce McGowens", "Jahmir Young", "Jalen Bridges",
"Kevin McCullar Jr.", "Trentyn Flowers", "Emanuel Miller", "Daniss Jenkins", "Tolu Smith",
"Anton Watson", "Adama Sanogo", "Malevy Leons", "Taze Moore", "Isaiah Mobley",
"Ron Harper Jr.", "Yuri Collins", "Tristen Newton", "Daishen Nix", "Skal Labissiere",
"Braxton Key", "Mac McClung", "Quincy Olivari", "Cui Yongxi", "Terence Davis",
"Jalen McDaniels", "Riley Minix", "Terry Taylor", "Isaiah Stevens", "James Wiseman",
"Jesse Edwards", "Alondes Williams", "Jahlil Okafor", "Zyon Pullin",

]

ARCHIVO_ALINEACIONES = "alineaciones.csv"
ARCHIVO_RESUMEN = "resumen_cruce_jugador_defensor.csv"

df = pd.read_csv(ARCHIVO_ALINEACIONES)
resultados = []

def minutos_a_decimal(min_str):
    if pd.isna(min_str):
        return 0
    try:
        partes = str(min_str).split(":")
        return int(partes[0]) + int(partes[1])/60
    except:
        return 0

for jugador in JUGADORES:
    print(f"Analizando {jugador}...")
    partidos_jugador = df[df['PLAYER_NAME'] == jugador]
    for _, row in partidos_jugador.iterrows():
        game_id = row['GAME_ID']
        team = row['TEAM']
        pos = row['START_POSITION']
        # Ahora toma TODOS los titulares rivales de la misma posición
        rivales = df[(df['GAME_ID'] == game_id) & (df['TEAM'] != team) & (df['START_POSITION'] == pos)]
        for _, rival_row in rivales.iterrows():
            defensor = rival_row['PLAYER_NAME']
            resultados.append({
                "Jugador": jugador,
                "Defensa": defensor,
                "MIN": row['MIN'],
                "PTS": row['PTS'],
                "REB": row['REB'],
                "AST": row['AST'],
                "FGM": row['FGM'],
                "FGA": row['FGA']
            })

df_res = pd.DataFrame(resultados)
if df_res.empty:
    print("No se han encontrado enfrentamientos para los jugadores dados.")
else:
    # Convertir MIN a minutos decimales
    df_res['MIN'] = df_res['MIN'].apply(minutos_a_decimal)

    # Filtro: solo emparejamientos con al menos 2 partidos jugados (puedes cambiar a >=1 si quieres verlos todos)
    cuenta_duelos = df_res.groupby(['Jugador', 'Defensa']).size().reset_index(name='Partidos')
    emparejamientos_validos = cuenta_duelos[cuenta_duelos['Partidos'] >= 2][['Jugador', 'Defensa']]
    df_res = pd.merge(df_res, emparejamientos_validos, on=['Jugador', 'Defensa'])

    resumen = df_res.groupby(['Jugador', 'Defensa']).agg({
        'MIN': 'mean',
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'FGM': 'sum',
        'FGA': 'sum'
    }).reset_index()
    resumen['%FG Media'] = (resumen['FGM'] / resumen['FGA']).round(3)
    resumen = resumen.rename(columns={
        'MIN': 'Media Min',
        'PTS': 'Media PTS',
        'REB': 'Media REB',
        'AST': 'Media AST'
    })
    resumen = resumen[['Jugador', 'Defensa', 'Media Min', 'Media PTS', 'Media REB', 'Media AST', '%FG Media']]
    resumen.to_csv(ARCHIVO_RESUMEN, index=False)
    print(f"\nArchivo guardado como: {ARCHIVO_RESUMEN}")




