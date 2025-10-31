import pandas as pd

# CSV file beolvasás
df = pd.read_excel("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_Validation.xlsx")

#A 'Accurate Passes' oszlop felbontása 3 oszlopra
df[['Successful Accurate Passes', 'All Accurate Passes', 'Accurate Pass Rating']] = df['Accurate Passes'].str.extract(r'(\d+)/(\d+) \((\d+)%\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket 
df['Successful Accurate Passes'] = df['Successful Accurate Passes'].fillna(0).astype(int)
df['All Accurate Passes'] = df['All Accurate Passes'].fillna(0).astype(int)
df['Accurate Pass Rating'] = df['Accurate Pass Rating'].fillna(0).astype(float)

# Az 'Accurate Passes' oszlop törlése
df.drop(columns=['Accurate Passes'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

#Extract 'Duels' and 'Duels won' from 'Duels (won)'
# A 'Duels (won)' oszlop felbontása 'Duels' és 'Duels won' oszlopokra
df[['Duels', 'Duels won']] = df['Duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Duels'] = df['Duels'].fillna(0).astype(int)
df['Duels won'] = df['Duels won'].fillna(0).astype(int)

#A 'Duels (won)' oszlop törlése
df.drop(columns=['Duels (won)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Ground Duels', 'Ground Duels won']] = df['Ground duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Ground Duels'] = df['Ground Duels'].fillna(0).astype(int)
df['Ground Duels won'] = df['Ground Duels won'].fillna(0).astype(int)

df.drop(columns=['Ground duels (won)'], inplace=True)

#///////////////////////////////////////////////////////////////////////////////////////////

df[['Aerial Duels', 'Aerial Duels won']] = df['Aerial duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Aerial Duels'] = df['Aerial Duels'].fillna(0).astype(int)
df['Aerial Duels won'] = df['Aerial Duels won'].fillna(0).astype(int)

df.drop(columns=['Aerial duels (won)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Dribble attempts', 'Succesfull Dribble attempts']] = df['Dribble attempts (succ.)'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Dribble attempts'] = df['Dribble attempts'].fillna(0).astype(int)
df['Succesfull Dribble attemps'] = df['Succesfull Dribble attempts'].fillna(0).astype(int)

df.drop(columns=['Dribble attempts (succ.)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Crosses attempts', 'Succesfull Crosses']] = df['Crosses'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Crosses attempts'] = df['Crosses attempts'].fillna(0).astype(int)
df['Succesfull Crosses'] = df['Succesfull Crosses'].fillna(0).astype(int)

df.drop(columns=['Crosses'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Long balls attempts', 'Succesfull Long balls']] = df['Long balls'].str.extract(r'(\d+) \((\d+)\)')

# A hiányzó értékek kitöltése 0-val, mielőtt integer-ré alakítsuk őket
df['Long balls attempts'] = df['Long balls attempts'].fillna(0).astype(int)
df['Succesfull Long balls'] = df['Succesfull Long balls'].fillna(0).astype(int)

df.drop(columns=['Long balls'], inplace=True)

# A frissített DataFrame elmentése új fájlba
df.to_excel("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_Validation.xlsx", index=False)

# Ismét betölteni DataFrame-et
df = pd.read_excel("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_Validation.xlsx")

# Egyedi játékosok listázása
players = df["Player"].unique()

# Kezdetben minden játékosnak létrehozni egy oszlopot, 0-ás kezdeti értékkel
for player in players:
    df[player] = 0

# Értékek kitöltése soronként
for index, row in df.iterrows():
    match_id = int(row["MatchID"])
    player = row["Player"]
    team = row["Team"]
    
    # Adott sor feltöltése 1, -1, 2, -2 stb értékekkel a MatchID-ra alapozva
    value = match_id if list(df[(df["MatchID"] == match_id)]["Team"].unique()).index(team) == 0 else -match_id
    df.at[index, player] = value

# Eredmény mentése új állományba
output_path_single_value = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_Validation1.xlsx"
df.to_excel(output_path_single_value, index=False)

df = pd.read_excel("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/test_Validation1.xlsx", engine = "openpyxl")

# Mentés CSV formátumba UTF-8 kódolással
df.to_csv("test_Validation1.csv", index=False, encoding="utf-8-sig")
