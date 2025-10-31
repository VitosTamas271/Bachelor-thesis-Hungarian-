import pandas as pd

# Load the CSV file with a specified encoding
df = pd.read_csv("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas2/Adatok/AllStatistics.csv", encoding='ISO-8859-1')

print(df.head(15))

#Split the 'Accurate Passes' column into three new columns
df[['Accurate Passes', 'All Passes Attempts', 'Accurate Pass Rating']] = df['Accurate Passes'].str.extract(r'(\d+)/(\d+) \((\d+)%\)')

# Fill NaN values with 0 before converting to integers
df['Accurate Passes'] = df['Accurate Passes'].fillna(0).astype(int)
df['All Passes Attempts'] = df['All Passes Attempts'].fillna(0).astype(int)
df['Accurate Pass Rating'] = df['Accurate Pass Rating'].fillna(0).astype(float)

# Drop the old 'Accurate Passes' column
df.drop(columns=['Accurate Passes'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

#Extract 'Duels' and 'Duels won' from 'Duels (won)'
df[['Duels', 'Duels won']] = df['Duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Duels'] = df['Duels'].fillna(0).astype(int)
df['Duels won'] = df['Duels won'].fillna(0).astype(int)

# Drop the old 'Duels (won)' column
df.drop(columns=['Duels (won)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Ground Duels', 'Ground Duels won']] = df['Ground duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Ground Duels'] = df['Ground Duels'].fillna(0).astype(int)
df['Ground Duels won'] = df['Ground Duels won'].fillna(0).astype(int)

# Drop the old 'Ground Duels (won)' column
df.drop(columns=['Ground duels (won)'], inplace=True)

#///////////////////////////////////////////////////////////////////////////////////////////

df[['Aerial Duels', 'Aerial Duels won']] = df['Aerial duels (won)'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Aerial Duels'] = df['Aerial Duels'].fillna(0).astype(int)
df['Aerial Duels won'] = df['Aerial Duels won'].fillna(0).astype(int)

# Drop the old 'Aerial Duels (won)' column
df.drop(columns=['Aerial duels (won)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Dribble attempts', 'Succesfull Dribble attempts']] = df['Dribble attempts (succ.)'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Dribble attempts'] = df['Dribble attempts'].fillna(0).astype(int)
df['Succesfull Dribble attemps'] = df['Succesfull Dribble attempts'].fillna(0).astype(int)

# Drop the old 'Dribble attempts (succ.)' column
df.drop(columns=['Dribble attempts (succ.)'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Crosses attempts', 'Succesfull Crosses']] = df['Crosses'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Crosses attempts'] = df['Crosses attempts'].fillna(0).astype(int)
df['Succesfull Crosses'] = df['Succesfull Crosses'].fillna(0).astype(int)

# Drop the old 'Crosses' column
df.drop(columns=['Crosses'], inplace=True)

#//////////////////////////////////////////////////////////////////////////////////////////

df[['Long balls attempts', 'Succesfull Long balls']] = df['Long balls'].str.extract(r'(\d+) \((\d+)\)')

# Fill NaN values with 0 and infer the correct data types
df['Long balls attempts'] = df['Long balls attempts'].fillna(0).astype(int)
df['Succesfull Long balls'] = df['Succesfull Long balls'].fillna(0).astype(int)

# Drop the old 'Long balls' column
df.drop(columns=['Long balls'], inplace=True)

# Save the updated DataFrame back to a CSV file
df.to_csv("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/AllStatistics.csv", index=False)

# Újra betöltjük az eredeti fájlt, hogy tiszta lappal induljunk
df = pd.read_csv("C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/AllStatistics.csv", encoding="ISO-8859-1")

# Egyedi játékosok listázása
players = df["Player"].unique()

# Kezdetben minden játékoshoz létrehozunk egy oszlopot, érték 0
for player in players:
    df[player] = 0

# Kitöltjük az értékeket soronként
for index, row in df.iterrows():
    match_id = int(row["MatchID"])
    player = row["Player"]
    team = row["Team"]
    
    # Adott sorhoz 1, -1, 2, -2 stb. MatchID alapján
    value = match_id if list(df[(df["MatchID"] == match_id)]["Team"].unique()).index(team) == 0 else -match_id
    df.at[index, player] = value

# Eredmény mentése új fájlba
output_path_single_value = "C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/Adatok/AllStatistics1.csv"
df.to_csv(output_path_single_value, index=False)
