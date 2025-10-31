import pandas as pd

# Fájl elérési útja
file_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_alapadat.txt"
excel_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_alapadat.xlsx"

# Oszlopnevek
columns = ["MatchID", "Team", "Player", "Goals", "Assists", "Total Tackles",
                 "Accurate Passes", "Duels (won)", "Ground Duels (won)", "Aerial Duels (won)",
                 "Minutes Played", "Position"]

# Adatok beolvasása és átalakítása
data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 12):  # Minden 11 sor egy játékos adatait tartalmazza
        matchid = lines[i].strip()
        team = lines[i+1].strip()
        player = lines[i+2].strip()
        goals = int(lines[i+3].strip())
        assists = int(lines[i+4].strip())
        total_tackles = int(lines[i+5].strip())
        accurate_passes = lines[i+6].strip()
        duels_won = lines[i+7].strip()
        ground_duels_won = lines[i+8].strip()
        aerial_duels_won = lines[i+9].strip()
        minutes_played = lines[i+10].strip()
        position = lines[i+11].strip()
        
        # Adatok hozzáadása a listához
        data.append([matchid, team, player, goals, assists, total_tackles, accurate_passes, 
                     duels_won, ground_duels_won, aerial_duels_won, minutes_played, position])

# DataFrame létrehozása és mentése Excelbe
df = pd.DataFrame(data, columns=columns)
df.to_excel(excel_path, index=False)

print(f"Excel fájl sikeresen létrehozva: {excel_path}")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd

# Fájl elérési útja
file_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_attackadat.txt"
excel_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_attackadat.xlsx"

# Oszlopnevek
columns = ["Team", "Player", "Shots on target", "Expected goals (xG)", "Shots off target", 
           "Shots blocked", "Dribble attempts (succ.)", "Notes", "Position"]

# Adatok beolvasása és átalakítása
data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 9):  # Minden 9 sor egy játékos adatait tartalmazza
        team = lines[i].strip()
        player = lines[i+1].strip()
        shotsontarget = lines[i+2].strip() #if lines[i+2].strip() != "-" else 0
        xG = lines[i+3].strip().replace("-", "0")
        shotsofftarget = lines[i+4].strip() #if lines[i+4].strip() != "-" else 0
        shotsblocked = lines[i+5].strip() #.replace("-", "0")
        dribbles = lines[i+6].strip() #.replace("-", "0")
        notes = lines[i+7].strip()
        position = lines[i+8].strip()

        # Adatok hozzáadása a listához
        data.append([team, player, shotsontarget, xG, shotsofftarget, shotsblocked, dribbles, notes, position])

# DataFrame létrehozása és mentése Excelbe
df = pd.DataFrame(data, columns=columns)
df.to_excel(excel_path, index=False)

print(f"Excel fájl sikeresen létrehozva: {excel_path}")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd

# Fájl elérési útja
file_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_defadat.txt"
excel_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_defadat.xlsx"

# Oszlopnevek
columns = ["Team", "Player", "Defensive actions", "Clearances", "Blocked shots", 
           "Interceptions", "Total tackels", "Dribbled past", "Defence notes", "Position"]

# Adatok beolvasása és átalakítása
data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 10):  # Minden 10 sor egy játékos adatait tartalmazza
        team = lines[i].strip()
        player = lines[i+1].strip()
        defactions = lines[i+2].strip() #if lines[i+2].strip() != "-" else 0
        clearances = lines[i+3].strip()
        blockedshots = lines[i+4].strip() #if lines[i+4].strip() != "-" else 0
        interceptions = lines[i+5].strip()
        tackels = lines[i+6].strip()
        dribbledpast = lines[i+7].strip()
        defnotes = lines[i+8].strip()
        position = lines[i+9].strip()

        # Adatok hozzáadása a listához
        data.append([team, player, defactions, clearances, blockedshots, interceptions, tackels, dribbledpast, defnotes, position])

# DataFrame létrehozása és mentése Excelbe
df = pd.DataFrame(data, columns=columns)
df.to_excel(excel_path, index=False)

print(f"Excel fájl sikeresen létrehozva: {excel_path}")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd

# Fájl elérési útja
file_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_passadat.txt"
excel_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_passadat.xlsx"

# Oszlopnevek
columns = ["Team", "Player", "Touches", "Accurate passes", "Key passes", "Crosses", "Long balls", "Pass notes", "Position"]

# Adatok beolvasása és átalakítása
data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 9):  # Minden 9 sor egy játékos adatait tartalmazza
        team = lines[i].strip()
        player = lines[i+1].strip()
        touches = lines[i+2].strip() #if lines[i+2].strip() != "-" else 0
        accpasses = lines[i+3].strip()
        keypasses = lines[i+4].strip() #if lines[i+4].strip() != "-" else 0
        crosses = lines[i+5].strip()
        longballs = lines[i+6].strip()
        passnotes = lines[i+7].strip()
        position = lines[i+8].strip()

        # Adatok hozzáadása a listához
        data.append([team, player, touches, accpasses, keypasses, crosses, longballs, passnotes, position])

# DataFrame létrehozása és mentése Excelbe
df = pd.DataFrame(data, columns=columns)
df.to_excel(excel_path, index=False)

print(f"Excel fájl sikeresen létrehozva: {excel_path}")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd

# Fájl elérési útja
file_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_duelsadat.txt"
excel_path = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_duelsadat.xlsx"

# Oszlopnevek
columns = ["Team", "Player", "Duels (won)", "Ground duels (won)", "Aerial duels (won)", "Possession lost", "Fouls", "Was fouled", "Offsides", "Position"]#, "Rating"]

# Adatok beolvasása és átalakítása
data = []
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 10):  # Minden 11 sor egy játékos adatait tartalmazza
        team = lines[i].strip()
        player = lines[i+1].strip()
        duelswon = lines[i+2].strip() #if lines[i+2].strip() != "-" else 0
        groundduelswon = lines[i+3].strip()
        aerialduelswon = lines[i+4].strip() #if lines[i+4].strip() != "-" else 0
        possloss = lines[i+5].strip()
        fouls = lines[i+6].strip()
        wasfouled = lines[i+7].strip()
        offsides = lines[i+8].strip()
        position = lines[i+9].strip()
        #rating = lines[i+10].strip()

        # Adatok hozzáadása a listához
        data.append([team, player, duelswon, groundduelswon, aerialduelswon, possloss, fouls, wasfouled, offsides, position])#, rating])

# DataFrame létrehozása és mentése Excelbe
df = pd.DataFrame(data, columns=columns)
df.to_excel(excel_path, index=False)

print(f"Excel fájl sikeresen létrehozva: {excel_path}")