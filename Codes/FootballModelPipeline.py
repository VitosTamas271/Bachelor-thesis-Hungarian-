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


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# PCA ANALYSIS

agregatedData_copy = agregatedData.drop(columns = "Player")

# Célváltozó (Rating) kivétele, csak bemeneti jellemzők megtartása
y = agregatedData_copy.drop(columns = ["Rating"])

# Adatok skálázása
scaler = StandardScaler()
scaled_y = scaler.fit_transform(y)

# PCA alkalmazása
pca = PCA()
pcaComponents = pca.fit_transform(scaled_y)

# Magyarázott variancia arány (Explained variance ratio)
explainedVariance = pca.explained_variance_ratio_
cumulativeVariance = np.cumsum(explainedVariance)

# Variancia diagram
plt.figure(figsize = (15, 10))
plt.plot(range(1, len(cumulativeVariance) + 1), cumulativeVariance, marker = 'o', linestyle = '--')
plt.title('Kumulált magyarázott variancia a főkomponensek számának a függvényében')
plt.xlabel('Kumulált magyarázott variancia')
plt.grid(True)
plt.tight_layout()
plt.show()

# Eredmény: első X komponens magyarázott varianciája
pcaVarianceDF = pd.DataFrame({
    "Főkomponens ": [f"PC{i + 1}" for i in range(len(explainedVariance))],
    "Magyarázott variancia (%) ": (explainedVariance * 100).round(2),
    "Kumulált variancia (%) ": (cumulativeVariance * 100).round(2)
})

#LINEÁRIS REGRESSZIÓ PCA ANALYSIS-EL

# Megnézzük az R² értéket különböző komponensszámok mellett (1-től 80-ig) Ebben nincsenek benne a pozíciók

r2_scores = []
componentRange = range(1, 81)
for n in componentRange:
    X_pca_n = pcaComponents[:, :n]
    X_test_n, X_train_n, y_test_n, y_train_n = train_test_split(X_pca_n, scaled_y, test_size = 0.1, random_state = 42)
    model_n = LinearRegression()
    model_n.fit(X_train_n, y_train_n)
    y_pred_n = model_n.predict(X_test_n)
    r2_scores.append(r2_score(y_test_n, y_pred_n))

# Ábra az R² érték alakulásáról
plt.figure(figsize = (15, 10))
plt.plot(componentRange, r2_scores, marker = 'o')
plt.title("Az R² érték alakulása a PCA komponensek számának a függvényében")
plt.xlabel("PCA komponensek száma")
plt.ylabel("Determinációs együttható (R²)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bemeneti változók: PCA komponensek
X_pca  = pcaComponents[:, :80]

X_train, X_test, y_train, y_test = train_test_split(X_pca, scaled_y, test_size = 0.1, random_state = 42)

pcaLinModel = LinearRegression()
pcaLinModel.fit(X_train, y_train)
pcaLinModelPredict = pcaLinModel.predict(X_test)

r2PCALinModel = r2_score(y_test, pcaLinModelPredict)
maePCALinModel = mean_absolute_error(y_test, pcaLinModelPredict)
msePCALinModel = mean_squared_error(y_test, pcaLinModelPredict)
mapePCALinModel = mean_absolute_percentage_error(y_test, pcaLinModelPredict)

print(f"Determinációs együttható (R²): {round(r2PCALinModel, 3)}")
print(f"Átlagos négyzetes hiba (MSE): {round(msePCALinModel, 3)}")
print(f"Átlagos abszolút hiba (MAE): {round(maePCALinModel, 3)}")
print(f"Mean absolute percentage error (MAPE): {round(mapePCALinModel, 3)}")

# Az első X komponenshez tartozó súlyok (komponens mátrix / loadings)
sulyok = pd.DataFrame(
    pcaComponents[:40].T,
    columns = [f"PC{i + 1}" for i in range(40)],
    index = y.columns
)

# Domináns változók minden komponenshez: abszolút érték alapján TOP 3
topSulyPerKomponens = {}
for komponens in sulyok.columns:
    topSulyok = sulyok[komponens].abs().sort_values(ascending = False).head(3).index.tolist()
    topSulyPerKomponens[komponens] = topSulyok

# Minden változóhoz kiszámítjuk az összes komponenshez tartozó abszolút súlyok összegét
variableImportance = sulyok.abs().sum(axis = 1).sort_values(ascending = False)

# Legfontosabb változók sorrendben (TOP X)
legfontosabbValtozok = variableImportance.head(20)
# print(legfontosabbValtozok)

# POSZT SZERINTI PCA ELEMZÉS

# Pozíció szerinti szétválogatás
defenderPlayers = agregatedData_copy[agregatedData_copy["Position_D"] == 1]
defenderPlayers = defenderPlayers.drop(columns = ["Position_D", "Position_M", "Position_F"])
midfielderPlayers = agregatedData_copy[agregatedData_copy["Position_M"] == 1]
midfielderPlayers = midfielderPlayers.drop(columns = ["Position_D", "Position_M", "Position_F"])
forwardPlayers = agregatedData_copy[agregatedData_copy["Position_F"] == 1]
forwardPlayers = forwardPlayers.drop(columns = ["Position_D", "Position_M", "Position_F"])

# Minden posztra kiválasztjuk a Rating (teljesítmény) értékeket
positionRatings = {
    "Defenders": defenderPlayers["Rating"],
    "Midfielders": midfielderPlayers["Rating"],
    "Forwards": forwardPlayers["Rating"]
}

# Boxplot a pozíciónkénti teljesítmény-eloszlásról
plt.figure(figsize = (15, 10))
plt.boxplot(positionRatings.values(), labels = positionRatings.keys(), patch_artist = True)
plt.title("Játékosok teljesítményének (Rating) eloszlása pozíciónként")
plt.ylabel("Rating")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.pipeline import Pipeline

# Egy függvény, amely PCA-t és lineáris regressziót végez egy adott poszt adataira
def PCAAnalysisByPosition(data, positionName):
    X = data.drop(columns = ["Rating"])
    y = data["Rating"]

    # Standardizálás + PCA + Lineáris regresszió pipeline-ban
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 80)),
        ('regressor', LinearRegression())
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
    pipeline.fit(X_train, y_train)
    y_pred_pca_pos = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred_pca_pos)
    mse = mean_squared_error(y_test, y_pred_pca_pos)
    mae = mean_absolute_error(y_test, y_pred_pca_pos)
    mape = mean_absolute_percentage_error(y_test, y_pred_pca_pos)

    return {
        "Pozíció ": positionName,
        "R²": round(r2, 3),
        "MSE": round(mse, 3),
        "MAE": round(mae, 3),
        "MAPE": round(mape, 3)
    }
# Modellek futtatása posztonként
eredmenyek = []
eredmenyek.append(PCAAnalysisByPosition(defenderPlayers, "Defenders"))
eredmenyek.append(PCAAnalysisByPosition(midfielderPlayers, "Midfielders"))
eredmenyek.append(PCAAnalysisByPosition(forwardPlayers, "Forwards"))

eredmenyekDF = pd.DataFrame(eredmenyek)
# print(eredmenyekDF)

# Függvény: Legdominánsabb változók egy adott pozíció első N PCA-komponensében
def legfontosabbValtozokPozicionkent(data, positionName, n_component = 5, top_n_features = 5):
    X = data.drop(columns = "Rating")

    # Standardizálás
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA illesztés
    pca = PCA(n_components = n_component)
    pca.fit(X_scaled)

    # Komponens súlyok
    sulyok = pd.DataFrame(
        pca.components_.T,
        columns = [f"PC{i + 1}" for i in range(n_component)],
        index = X.columns
    )

    # Minden komponenshez kiválasztjuk a legnagyobb abszolút értékű változókat
    legfontosabbStatisztikak = {}
    for suly in sulyok.columns:
        top = sulyok[suly].abs().sort_values(ascending = False).head(top_n_features).index.tolist()
        legfontosabbStatisztikak[suly] = top
    
    return pd.DataFrame(legfontosabbStatisztikak)

# Futtatás posztonként
legfontosabbVedoStatisztikak = legfontosabbValtozokPozicionkent(defenderPlayers, "Defenders")
legfontosabbKozeppalyasStatisztikak = legfontosabbValtozokPozicionkent(midfielderPlayers, "Midfielders")
legfontosabbTamadoStatisztikak = legfontosabbValtozokPozicionkent(forwardPlayers, "Forwards")

print(legfontosabbVedoStatisztikak)
print(legfontosabbKozeppalyasStatisztikak)
print(legfontosabbTamadoStatisztikak)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# RIDGE / LASSO / ELASTIC NET REGRESSION

exclude_columns = ["Player", "MatchID", "Rating"]
features = [col for col in agregatedData.columns if col not in exclude_columns]

X = agregatedData[features]
y = agregatedData["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha = 1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print("Ridge MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("Ridge R2:", r2_score(y_test, y_pred_ridge))
print("Ridge MAPE:", mean_absolute_percentage_error(y_test, y_pred_ridge))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print("Lasso MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("Lasso R2:", r2_score(y_test, y_pred_lasso))
print("Lasso MAPE:", mean_absolute_percentage_error(y_test, y_pred_lasso))

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
y_pred_elastic = elastic.predict(X_test_scaled)
print("ElasticNet MAE:", mean_absolute_error(y_test, y_pred_elastic))
print("ElasticNet R2:", r2_score(y_test, y_pred_elastic))
print("ElasticNet MAPE:", mean_absolute_percentage_error(y_test, y_pred_elastic))


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# RANDOM FOREST

targetColumn = "Rating"
excludeColumns = ["Player", "MatchID", targetColumn]
featureColumns = [col for col in agregatedData.columns if col not in excludeColumns]

X = agregatedData[featureColumns]
y = agregatedData[targetColumn]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

randomForestModel = RandomForestRegressor(n_estimators = 200, random_state = 42)
randomForestModel.fit(X_train, y_train)

y_pred = randomForestModel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE_RandomForest: {mae: .3f}, R2_RandomForest: {r2: .3f}, MAPE_RandomForest: {mape: .3f}")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# XGBOOOST

# Csak numerikus típusú oszlopokat tartunk meg a feature setben
exclude_columns = ["Player", "MatchID", "Rating"]
feature_columns = [
    col for col in agregatedData.columns
    if col not in exclude_columns and agregatedData[col].dtype in [np.int64, np.float64, np.bool_]
]

X = agregatedData[feature_columns]
y = agregatedData["Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# XGBoost modell létrehozása és tanítása
xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=1,
    colsample_bytree=0.7,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Előrejelzés és kiértékelés
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)

print(f"MAE_XGB: {mae_xgb: .3f}, R2_XGB: {r2_xgb: .3f}, MAPE_XGB: {mape_xgb: .3f}")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# POZíCIÓ SPECIFIKUS MODELLEK

positionResults = {}

for position in ['D', 'M', 'F']:
    positionColumn = f'Position_{position}'
    if positionColumn in agregatedData.columns:
        print(f"\n--- {position} pozíció modellezése ---")

        # Adatszűrés
        positionData = agregatedData[agregatedData[positionColumn] == 1]
        positionFeatureColumns = [col for col in positionData.columns if col not in ["Player", "MatchID", "Rating"] and positionData[col].dtype in [np.int64, np.float64, np.bool_]]
        X = positionData[positionFeatureColumns]
        y = positionData["Rating"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Skálázás a regressziókhoz
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 100-Fold Cross-Validation
        cv = KFold(n_splits = 60, shuffle = True, random_state = 42)

        positionModelScores = {}

        # Ridge Regression
        ridge = Ridge(alpha = 1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pos_pred_ridge = ridge.predict(X_test_scaled)
        mae_cv = -cross_val_score(ridge, X_train_scaled, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
        r2_cv = cross_val_score(ridge, X_train_scaled, y_train, cv=cv, scoring='r2').mean()
        positionModelScores["Ridge"] = {
            "MAE_Ridge": round(mean_absolute_error(y_test, y_pos_pred_ridge), 3),
            "R2_Ridge": round(r2_score(y_test, y_pos_pred_ridge), 3),
            "MAPE (%)": round(mean_absolute_percentage_error(y_test, y_pos_pred_ridge) * 100, 3),
            "CV MAE": round(mae_cv, 3),
            "CV R2": round(r2_cv, 3)
        }

        # Lasso Regression
        lasso = Lasso(alpha = 0.1)
        lasso.fit(X_train_scaled, y_train)
        y_pos_pred_lasso = lasso.predict(X_test_scaled)
        mae_cv = -cross_val_score(lasso, X_train_scaled, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
        r2_cv = cross_val_score(lasso, X_train_scaled, y_train, cv=cv, scoring='r2').mean()
        positionModelScores["Lasso"] = {
            "MAE_Lasso": round(mean_absolute_error(y_test, y_pos_pred_lasso), 3),
            "R2_Lasso": round(r2_score(y_test, y_pos_pred_lasso), 3),
            "MAPE (%)": round(mean_absolute_percentage_error(y_test, y_pos_pred_lasso) * 100, 3),
            "CV MAE": round(mae_cv, 3),
            "CV R2": round(r2_cv, 3)
        }

        # ElasticNet Regression
        elastic = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
        elastic.fit(X_train_scaled, y_train)
        y_pos_pred_elastic = elastic.predict(X_test_scaled)
        mae_cv = -cross_val_score(elastic, X_train_scaled, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
        r2_cv = cross_val_score(elastic, X_train_scaled, y_train, cv=cv, scoring='r2').mean()
        positionModelScores["ElasticNet"] = {
            "MAE_ElasticNet": round(mean_absolute_error(y_test, y_pos_pred_elastic), 3),
            "R2_ElasticNet": round(r2_score(y_test, y_pos_pred_elastic), 3),
            "MAPE (%)": round(mean_absolute_percentage_error(y_test, y_pos_pred_elastic) * 100, 3),
            "CV MAE": round(mae_cv, 3),
            "CV R2": round(r2_cv, 3)
        }

        # Random Forest
        randomforest = RandomForestRegressor(n_estimators = 300, random_state = 42)
        randomforest.fit(X_train, y_train)
        y_pos_pred_randfor = randomforest.predict(X_test)
        mae_cv = -cross_val_score(randomforest, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
        r2_cv = cross_val_score(randomforest, X_train, y_train, cv=cv, scoring='r2').mean()
        positionModelScores["Random Forest"] = {
            "MAE_RandomForest": round(mean_absolute_error(y_test, y_pos_pred_randfor), 3),
            "R2_RandomForest": round(r2_score(y_test, y_pos_pred_randfor), 3),
            "MAPE (%)": round(mean_absolute_percentage_error(y_test, y_pos_pred_randfor) * 100, 3),
            "CV MAE": round(mae_cv, 3),
            "CV R2": round(r2_cv, 3)
        }

        # XGBoost
        xgboost = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=1,
            colsample_bytree=0.7,
            random_state=42
        )
        xgboost.fit(X_train, y_train)
        y_pos_pred_xgboost = xgboost.predict(X_test)
        mae_cv = -cross_val_score(xgboost, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error').mean()
        r2_cv = cross_val_score(xgboost, X_train, y_train, cv=cv, scoring='r2').mean()
        positionModelScores["XGBoost"] = {
            "MAE_XGBoost": round(mean_absolute_error(y_test, y_pos_pred_xgboost), 3),
            "R2_XGBoost": round(r2_score(y_test, y_pos_pred_xgboost), 3),
            "MAPE (%)": round(mean_absolute_percentage_error(y_test, y_pos_pred_xgboost) * 100, 3),
            "CV MAE": round(mae_cv, 3),
            "CV R2": round(r2_cv, 3)
        }

        positionResults[position] = positionModelScores

pd.set_option('display.max_rows', None)       # összes sor
pd.set_option('display.max_columns', None)    # összes oszlop
pd.set_option('display.width', None)          # ne tördeljen
pd.set_option('display.max_colwidth', None)   # teljes szövegek megjelenítése

positionResultsDF = pd.DataFrame(positionResults).T
print(positionResultsDF)

importances = pd.Series(xgb_model.feature_importances_, index = feature_columns)
top20Features = importances.sort_values(ascending = False).head(20)

plt.figure(figsize = (15, 10))
top20Features.plot(kind = "barh")
plt.title("Top 20 legfontosabb jellemző - XGBoost Feature Importance")
plt.xlabel("Fontosság")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
plt.title("Top 20 legfontosabb jellemző az XGBoost modellben")
plt.tight_layout()
plt.show()

param_grid = {
    'n_estimators': [200, 300, 400, 500, 600],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'colsample_bytree': [0.7, 0.8, 0.9, 1]
}

grid = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Legjobb paraméterek:", grid.best_params_)

params = {
    'n_estimators': [200, 300, 400, 500, 600],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'colsample_bytree': [0.7, 0.8, 0.9, 1]
}

grid = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_distributions = params,
    n_iter = 50,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Legjobb parameterek: ", grid.best_params_)