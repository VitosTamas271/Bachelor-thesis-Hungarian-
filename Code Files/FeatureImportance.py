import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score,KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Adatok beolvasása az állományból
data = pd.read_csv('C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/AllStatistics1.csv', encoding='ISO-8859-1')

# A "Minute Played" oszlopból eltávolítani a "'" karaktert, és átkonvertálni "int"-be 
data['Minutes Played'] = data['Minutes Played'].str.replace("'", "").astype('int64')

# A kategorikus változók numerikussá alakítása One-Hot Encoding-el
data = pd.get_dummies(data, columns=["Position"], dtype=int)

# Új statisztikai adatok behozatala
data["Duel Win Rate"] = (data["Duels won"] / data["Duels"]).round(2)
data["Ground Duel Win Rate"] = (data["Ground Duels won"] / data["Ground Duels"]).round(2)
data["Aerial Duel Win Rate"] = (data["Aerial Duels won"] / data["Aerial Duels"]).round(2)
data["Tackel Success Rate"] = (data["Total tackels"] / (data["Total tackels"] + data["Dribbled past"])).round(2)
data["Possession Loss Rate"] = (data["Possession lost"] / data["Touches"]).round(2)
data["Successful Long Ball Rate"] = (data["Succesfull Long balls"] / data["Long balls attempts"]).round(2)
data["Successful Cross Rate"] = (data["Succesfull Crosses"] / data["Crosses attempts"]).round(2)
data["Interception Efficiency"] = (data["Interceptions"] / data["Defensive actions"]).round(2)
data["Clearance Efficiency"] = (data["Clearances"] / data["Defensive actions"]).round(2)
data["Foul Rate"] = (data["Fouls"] / data["Duels"]).round(2)

# Játékos oszlopok azonosítása
allPlayers = set(data["Player"].unique())
playerColumns = [col for col in data.columns if col in allPlayers]

print("Oszlopnevek:", data.columns.tolist())

statisticsColumns = data.columns.difference(playerColumns + ["MatchID", "Team", "Player", "Notes", "Defence notes", "Pass notes"])
statisticsColumns = [col for col in statisticsColumns if data[col].dtype in [int, float]]

# Meccsenként a játékosok szétválogatása csapatokra minden MatchID-re
teamInfoPerMatch = {}
allMatchIDs = data["MatchID"].unique()

for matchID in allMatchIDs:
    matchData = data[data["MatchID"] == matchID]
    team1 = set()
    team2 = set()
    playersInMatch = matchData["Player"].tolist()
    matchPlayerColumns = [col for col in data.columns if col in playersInMatch]

    for _, row in matchData.iterrows():
        playerName = row["Player"]
        for teamMate in matchPlayerColumns:
            value = row[teamMate]
            if value == matchID:
                team1.add(playerName)
            elif value == -matchID:
                team2.add(playerName)

    teamInfoPerMatch[matchID] = {
        "team1": sorted(team1),
        "team2": sorted(team2)
    }

# Aggregált csapatstatisztikák hozzárendelése minden játékoshoz
agregatedRows = []

for matchID in allMatchIDs:
    matchData = data[data["MatchID" ] == matchID]
    teams = teamInfoPerMatch[matchID]
    team1 = teams["team1"]
    team2 = teams["team2"]

    team1Statistics = matchData[matchData["Player"].isin(team1)][statisticsColumns]
    team2Statistics = matchData[matchData["Player"].isin(team2)][statisticsColumns]

    team1Summary = {
        "team_total_goals": team1Statistics["Goals"].sum(),
        "team_total_assists": team1Statistics["Assists"].sum(),
        "team_total_shots": team1Statistics["Shots on target"].sum(),
        "team_avg_xG": team1Statistics["Expected goals (xG)"].mean(),
        "team_total_shotsoff": team1Statistics["Shots off target"].sum(),
        "team_shots_blocked": team1Statistics["Shots blocked"].sum(),
        "team_deff_actions": team1Statistics["Defensive actions"].mean(),
        "team_clearances": team1Statistics["Clearances"].mean(),
        "team_blocked_shots": team1Statistics["Blocked shots"].sum(),
        "team_interceptions": team1Statistics["Interceptions"].sum(),
        "team_total_tackels": team1Statistics["Total tackels"].sum(),
        "team_dribbled_past": team1Statistics["Dribbled past"].sum(),
        "team_avg_touches": team1Statistics["Touches"].mean(),
        "team_avg_acc_pass_rating": team1Statistics["Accurate Pass Rating (%)"].mean(),
        "team_avg_duel_rate": team1Statistics["Duel Win Rate"].mean(),
        "team_avg_ground_deul_rate": team1Statistics["Ground Duel Win Rate"].mean(),
        "team_avg_aerial_duel_rate": team1Statistics["Aerial Duel Win Rate"].mean(),
        "team_possession_lost_rate": team1Statistics["Possession Loss Rate"].mean(),
        "team_avg_succ_long_balls": team1Statistics["Successful Long Ball Rate"].mean(),
        "team_avg_succ_cross_rate": team1Statistics["Successful Cross Rate"].mean(),
        "team_avg_fouls_rate": team1Statistics["Foul Rate"].mean(),
        "team_total_key_passes": team1Statistics["Key passes"].sum()
    }

    team2Summary = {
        "team_total_goals": team2Statistics["Goals"].sum(),
        "team_total_assists": team2Statistics["Assists"].sum(),
        "team_total_shots": team2Statistics["Shots on target"].sum(),
        "team_avg_xG": team2Statistics["Expected goals (xG)"].mean(),
        "team_total_shotsoff": team2Statistics["Shots off target"].sum(),
        "team_shots_blocked": team2Statistics["Shots blocked"].sum(),
        "team_deff_actions": team2Statistics["Defensive actions"].mean(),
        "team_clearances": team2Statistics["Clearances"].mean(),
        "team_blocked_shots": team2Statistics["Blocked shots"].sum(),
        "team_interceptions": team2Statistics["Interceptions"].sum(),
        "team_total_tackels": team2Statistics["Total tackels"].sum(),
        "team_dribbled_past": team2Statistics["Dribbled past"].sum(),
        "team_avg_touches": team2Statistics["Touches"].mean(),
        "team_avg_acc_pass_rating": team2Statistics["Accurate Pass Rating (%)"].mean(),
        "team_avg_duel_rate": team2Statistics["Duel Win Rate"].mean(),
        "team_avg_ground_deul_rate": team2Statistics["Ground Duel Win Rate"].mean(),
        "team_avg_aerial_duel_rate": team2Statistics["Aerial Duel Win Rate"].mean(),
        "team_possession_lost_rate": team2Statistics["Possession Loss Rate"].mean(),
        "team_avg_succ_long_balls": team2Statistics["Successful Long Ball Rate"].mean(),
        "team_avg_succ_cross_rate": team2Statistics["Successful Cross Rate"].mean(),
        "team_avg_fouls_rate": team2Statistics["Foul Rate"].mean(),
        "team_total_key_passes": team2Statistics["Key passes"].sum()
    }

    for _, row in matchData.iterrows():
        player = row["Player"]
        rowData = row[["Player", "MatchID"] + statisticsColumns].to_dict()

        if player in team1:
            rowData.update({
                "team_total_goals": team1Summary["team_total_goals"],
                "team_total_assists": team1Summary["team_total_assists"],
                "team_total_shots": team1Summary["team_total_shots"],
                "team_avg_xG": team1Summary["team_avg_xG"],
                "team_total_shotsoff": team1Summary["team_total_shotsoff"],
                "team_shots_blocked": team1Summary["team_shots_blocked"],
                "team_deff_actions": team1Summary["team_deff_actions"],
                "team_clearances": team1Summary["team_clearances"],
                "team_blocked_shots": team1Summary["team_blocked_shots"],
                "team_interceptions": team1Summary["team_interceptions"],
                "team_total_tackels": team1Summary["team_total_tackels"],
                "team_dribbled_past": team1Summary["team_dribbled_past"],
                "team_avg_touches": team1Summary["team_avg_touches"],
                "team_avg_acc_pass_rating": team1Summary["team_avg_acc_pass_rating"],
                "team_avg_duel_rate": team1Summary["team_avg_duel_rate"],
                "team_avg_ground_deul_rate": team1Summary["team_avg_ground_deul_rate"],
                "team_avg_aerial_duel_rate": team1Summary["team_avg_aerial_duel_rate"],
                "team_possession_lost_rate": team1Summary["team_possession_lost_rate"],
                "team_avg_succ_long_balls": team1Summary["team_avg_succ_long_balls"],
                "team_avg_succ_cross_rate": team1Summary["team_avg_succ_cross_rate"],
                "team_avg_fouls_rate": team1Summary["team_avg_fouls_rate"],
                "team_total_key_passes": team1Summary["team_total_key_passes"],
                "opp_team_total_goals": team2Summary["team_total_goals"],
                "opp_team_total_assists": team2Summary["team_total_assists"],
                "opp_team_total_shots": team2Summary["team_total_shots"],
                "opp_team_avg_xG": team2Summary["team_avg_xG"],
                "opp_team_total_shotsoff": team2Summary["team_total_shotsoff"],
                "opp_team_shots_blocked": team2Summary["team_shots_blocked"],
                "opp_team_deff_actions": team2Summary["team_deff_actions"],
                "opp_team_clearances": team2Summary["team_clearances"],
                "opp_team_blocked_shots": team2Summary["team_blocked_shots"],
                "opp_team_interceptions": team2Summary["team_interceptions"],
                "opp_team_total_tackels": team2Summary["team_total_tackels"],
                "opp_team_dribbled_past": team2Summary["team_dribbled_past"],
                "opp_team_avg_touches": team2Summary["team_avg_touches"],
                "opp_team_avg_acc_pass_rating": team2Summary["team_avg_acc_pass_rating"],
                "opp_team_avg_duel_rate": team2Summary["team_avg_duel_rate"],
                "opp_team_avg_ground_deul_rate": team2Summary["team_avg_ground_deul_rate"],
                "opp_team_avg_aerial_duel_rate": team2Summary["team_avg_aerial_duel_rate"],
                "opp_team_possession_lost_rate": team2Summary["team_possession_lost_rate"],
                "opp_team_avg_succ_long_balls": team2Summary["team_avg_succ_long_balls"],
                "opp_team_avg_succ_cross_rate": team2Summary["team_avg_succ_cross_rate"],
                "opp_team_avg_fouls_rate": team2Summary["team_avg_fouls_rate"],
                "opp_team_total_key_passes": team2Summary["team_total_key_passes"],
            })
        elif player in team2:
            rowData.update({
                "team_total_goals": team2Summary["team_total_goals"],
                "team_total_assists": team2Summary["team_total_assists"],
                "team_total_shots": team2Summary["team_total_shots"],
                "team_avg_xG": team2Summary["team_avg_xG"],
                "team_total_shotsoff": team2Summary["team_total_shotsoff"],
                "team_shots_blocked": team2Summary["team_shots_blocked"],
                "team_deff_actions": team2Summary["team_deff_actions"],
                "team_clearances": team2Summary["team_clearances"],
                "team_blocked_shots": team2Summary["team_blocked_shots"],
                "team_interceptions": team2Summary["team_interceptions"],
                "team_total_tackels": team2Summary["team_total_tackels"],
                "team_dribbled_past": team2Summary["team_dribbled_past"],
                "team_avg_touches": team2Summary["team_avg_touches"],
                "team_avg_acc_pass_rating": team2Summary["team_avg_acc_pass_rating"],
                "team_avg_duel_rate": team2Summary["team_avg_duel_rate"],
                "team_avg_ground_deul_rate": team2Summary["team_avg_ground_deul_rate"],
                "team_avg_aerial_duel_rate": team2Summary["team_avg_aerial_duel_rate"],
                "team_possession_lost_rate": team2Summary["team_possession_lost_rate"],
                "team_avg_succ_long_balls": team2Summary["team_avg_succ_long_balls"],
                "team_avg_succ_cross_rate": team2Summary["team_avg_succ_cross_rate"],
                "team_avg_fouls_rate": team2Summary["team_avg_fouls_rate"],
                "team_total_key_passes": team2Summary["team_total_key_passes"],
                "opp_team_total_goals": team1Summary["team_total_goals"],
                "opp_team_total_assists": team1Summary["team_total_assists"],
                "opp_team_total_shots": team1Summary["team_total_shots"],
                "opp_team_avg_xG": team1Summary["team_avg_xG"],
                "opp_team_total_shotsoff": team1Summary["team_total_shotsoff"],
                "opp_team_shots_blocked": team1Summary["team_shots_blocked"],
                "opp_team_deff_actions": team1Summary["team_deff_actions"],
                "opp_team_clearances": team1Summary["team_clearances"],
                "opp_team_blocked_shots": team1Summary["team_blocked_shots"],
                "opp_team_interceptions": team1Summary["team_interceptions"],
                "opp_team_total_tackels": team1Summary["team_total_tackels"],
                "opp_team_dribbled_past": team1Summary["team_dribbled_past"],
                "opp_team_avg_touches": team1Summary["team_avg_touches"],
                "opp_team_avg_acc_pass_rating": team1Summary["team_avg_acc_pass_rating"],
                "opp_team_avg_duel_rate": team1Summary["team_avg_duel_rate"],
                "opp_team_avg_ground_deul_rate": team1Summary["team_avg_ground_deul_rate"],
                "opp_team_avg_aerial_duel_rate": team1Summary["team_avg_aerial_duel_rate"],
                "opp_team_possession_lost_rate": team1Summary["team_possession_lost_rate"],
                "opp_team_avg_succ_long_balls": team1Summary["team_avg_succ_long_balls"],
                "opp_team_avg_succ_cross_rate": team1Summary["team_avg_succ_cross_rate"],
                "opp_team_avg_fouls_rate": team1Summary["team_avg_fouls_rate"],
                "opp_team_total_key_passes": team1Summary["team_total_key_passes"],
            })
        else:
            continue

        agregatedRows.append(rowData)

agregatedData = pd.DataFrame(agregatedRows)
agregatedData = agregatedData.fillna(0)

# Frissített változat: csak a célváltozó ("Rating") kizárása, minden más változó bevonása
X_full = agregatedData.drop(columns=["Rating"], errors='ignore')
y_full = agregatedData["Rating"]

# Csak numerikus adatokat tartunk meg (modellezéshez szükséges)
X_full = X_full.select_dtypes(include=[np.number])

# Skálázás
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)

# Modell tanítása
model_full = RandomForestRegressor(n_estimators=100, random_state=42)
model_full.fit(X_scaled_full, y_full)

# Fontosságok kiszámítása
importances_full = model_full.feature_importances_
indices_full = np.argsort(importances_full)[::-1]
feature_names_full = X_full.columns[indices_full]
importance_values_full = importances_full[indices_full]

# Táblázat létrehozása
full_importance_df = pd.DataFrame({
    "Feature": feature_names_full,
    "Importance": importance_values_full
})

# tools.display_dataframe_to_user(name="Minden Feature Importance - Rating", dataframe=full_importance_df)

full_importance_df.head(10)
