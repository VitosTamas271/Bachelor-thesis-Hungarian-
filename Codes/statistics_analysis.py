import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Adatok beolvasása az állományból
statisticData = pd.read_csv('C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/AllStatistics1.csv', encoding='ISO-8859-1')

# A "Minute Played" oszlopból eltávolítani a "'" karaktert, és átkonvertálni "int"-be 
statisticData['Minutes Played'] = statisticData['Minutes Played'].str.replace("'", "").astype('int64')

# Nem szükséges (nem fontos) adatokat tartalmazó oszlopok eltávolítása
statisticData.drop(columns = ["Team", "Player", "Notes", "Defence notes", "Pass notes"], inplace = True)

# A kategorikus változók numerikussá alakítása One-Hot Encoding-el
statisticData = pd.get_dummies(statisticData, columns=["Position"], dtype=int)

# Új statisztikai adatok létrehozása
statisticData["Pass Accuracy"] = (statisticData["Successful Accurate Passes"] / statisticData["All Accurate Passes"]).round(2)
statisticData["Duel Win Rate"] = (statisticData["Duels won"] / statisticData["Duels"]).round(2)
statisticData["Ground Duel Win Rate"] = (statisticData["Ground Duels won"] / statisticData["Ground Duels"]).round(2)
statisticData["Aerial Duel Win Rate"] = (statisticData["Aerial Duels won"] / statisticData["Aerial Duels"]).round(2)
statisticData["Tackle Success Rate"] = (statisticData["Total Tackles"] / (statisticData["Total Tackles"] + statisticData["Dribbled past"])).round(2)
statisticData["Possession Loss Rate"] = (statisticData["Possession lost"] / statisticData["Touches"]).round(2)
statisticData["Successful Long Ball Rate"] = (statisticData["Succesfull Long balls"] / statisticData["Long balls attempts"]).round(2)
statisticData["Successful Cross Rate"] = (statisticData["Succesfull Crosses"] / statisticData["Crosses attempts"]).round(2)
statisticData["Interception Efficiency"] = (statisticData["Interceptions"] / statisticData["Defensive actions"]).round(2)
statisticData["Clearance Efficiency"] = (statisticData["Clearances"] / statisticData["Defensive actions"]).round(2)
statisticData["Foul Rate"] = (statisticData["Fouls"] / statisticData["Duels"]).round(2)

# # Kicserélni a NaN értékeket 0.0-ra az egész DataFrame-ben
statisticData.fillna(0.0, inplace=True)

# Különválasztani a játékosokat posztok szerint az eredeti oszlop alapján
defenderPlayers = statisticData[statisticData["Position_D"] == 1]
defenderPlayers = defenderPlayers.drop(columns = ["Position_D", "Position_M", "Position_F"])
midfielderPlayers = statisticData[statisticData["Position_M"] == 1]
midfielderPlayers = midfielderPlayers.drop(columns = ["Position_D", "Position_M", "Position_F"])
forwardPlayers = statisticData[statisticData["Position_F"] == 1]
forwardPlayers = forwardPlayers.drop(columns = ["Position_M", "Position_D", "Position_F"])


# Releváns változók meghatározása minden posztra
defender_features = ["Successful Accurate Passes", "All Accurate Passes", "Duels", "Duels won", "Ground Duels", "Ground Duels won", "Aerial Duels", "Aerial Duels won", "Defensive actions", "Clearances", "Blocked shots", "Interceptions", "Dribbled past", "Total Tackles", "Touches", "Long balls attempts", "Succesfull Long balls", "Possession lost", "Fouls", "Minutes Played"]
midfielder_features = ["Assists", "Successful Accurate Passes", "All Accurate Passes", "Duels", "Duels won", "Ground Duels", "Ground Duels won", "Aerial Duels", "Aerial Duels won", "Shots on target", "Expected goals (xG)", "Dribble attempts", "Succesfull Dribble attempts", "Touches", "Key passes", "Long balls attempts", "Succesfull Long balls", "Possession lost", "Minutes Played"]
forward_features = ["Goals", "Assists", "Successful Accurate Passes", "All Accurate Passes", "Duels", "Duels won", "Ground Duels", "Ground Duels won", "Aerial Duels", "Aerial Duels won", "Shots on target", "Expected goals (xG)", "Shots off target", "Shots blocked", "Dribble attempts", "Succesfull Dribble attempts", "Touches", "Possession lost", "Was fouled", "Offsides", "Minutes Played"]

X_def = defenderPlayers.drop(columns = ["Rating"])
X_mid = midfielderPlayers.drop(columns = ["Rating"])
X_for = forwardPlayers.drop(columns = ["Rating"])

# Csak a numerikus változókat tartjuk meg a statisztikai elemzéshez
numeric_columns = statisticData.select_dtypes(include=['number']).columns
num_statisticData = statisticData[numeric_columns]
num_statisticData = num_statisticData.drop(columns = ["Position_D", "Position_M", "Position_F"])

def_cor_matrix = X_def.corr()
axis_corr = sns.heatmap(
    def_cor_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    square=False,  
    annot=True, 
    fmt=".2f",  
    annot_kws={"size": 10} 
)

plt.figure(figsize=(35, 30))
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()

mid_cor_matrix = X_mid.corr()
axis_corr = sns.heatmap(
    mid_cor_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    square=False,  
    annot=True, 
    fmt=".2f",  
    annot_kws={"size": 10} 
)

plt.figure(figsize=(35, 30))
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()

for_cor_matrix = X_for.corr()
axis_corr = sns.heatmap(
    for_cor_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    square=False,  
    annot=True, 
    fmt=".2f",  
    annot_kws={"size": 10} 
)

plt.figure(figsize=(35, 30))
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()

#//////////////////////////////////////////////////////////////////////////////////////////////////
#LEGFONTOSABB STATISZTIKÁK KIVÁLASZTÁSA

# Korrelációs mátrix kiszámítása, hogy aztán meg tudjuk határozni a legfontosabb statisztikákat
Corr_Matrix = num_statisticData.corr()

axis_corr = sns.heatmap(
    Corr_Matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    square=False,  
    annot=True, 
    fmt=".2f",  
    annot_kws={"size": 10} 
)

plt.figure(figsize=(35, 30))
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()

# Kiemeljük a "Rating" oszlop korrelációs értékeit és sorba rendezzük
rating_corr = Corr_Matrix["Rating"].sort_values(ascending = False)

# A 15 legnagyobb korrelációjú statisztika a "Rating" értékkel
plt.figure(figsize = (15, 10))
sns.barplot(x = rating_corr.head(15).index, y = rating_corr.head(15).values, hue = rating_corr.tail(15).index, palette = "Blues")
plt.xticks(rotation = 90)
plt.title("A 15 legnagyobb korrelációjú statisztika a Rating értékkel")
plt.xlabel("Statisztikai változók")
plt.ylabel("Korrelációs érték")
plt.show()

# A 15 legkisseb korrelációjú statisztika a "Rating" értékkel
plt.figure(figsize = (15, 10))
sns.barplot(x = rating_corr.tail(15).index, y = rating_corr.tail(15).values, hue = rating_corr.tail(15).index, palette = "Blues")
plt.xticks(rotation = 90)
plt.title("A 15 legkisebb korrelációjú statisztika a Rating értékkel")
plt.xlabel("Statisztikai változók")
plt.ylabel("Korrelációs érték")
plt.show()

# A célváltozó meghatározása
X = num_statisticData.drop(columns = ["Rating"])
y = num_statisticData["Rating"]

# Szétválasztani az adatokat tanuló és tesztelési halmazokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Adatok skálázása
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest modell betanítása a legfontosabb statisztikai változók kiválasztása érdekében
random_forest_modell = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_forest_modell.fit(X_train_scaled, y_train)

# Fontossági súlyok kinyerése
feature_importance = pd.Series(random_forest_modell.feature_importances_, index = X.columns).sort_values(ascending = False)

# Megjelenítés
plt.figure(figsize = (15, 10))
sns.barplot(x = feature_importance.head(15).index, y = feature_importance.head(15).values, hue = feature_importance.head(15).index, palette = "Reds")
plt.xticks(rotation = 90)
plt.title("A 15 legfontosabb statisztika a játékosok teljesítményének értékelésében")
plt.xlabel("Statisztikai változók")
plt.ylabel("Fontossági súly")
plt.show()

# Visszaadjuk az első 15 legfontosabb statisztikát
print(feature_importance.head(15))

# Alternatív módszerek a legfontosabb statisztikák kiválasztására
# XGBoost

# Modell betanítása
xgb_modell = xgb.XGBRegressor(n_estimators = 100, random_state = 42, eval_metric = 'rmse')
xgb_modell.fit(X_train_scaled, y_train)

# Feature importance XGBoost szerint
feature_importance_xgb = pd.Series(xgb_modell.feature_importances_, index = X.columns).sort_values(ascending = False)

# XGBoost Feature importance megjelenítés
plt.figure(figsize = (15, 10))
sns.barplot(x = feature_importance_xgb.head(15).index, y = feature_importance_xgb.head(15).values, hue = feature_importance_xgb.head(15), palette = "Purples")
plt.xticks(rotation = 90)
plt.title("A legfontosabb 15 statisztikai adat az XGBoost szerint")
plt.xlabel("Statisztikai változók")
plt.ylabel("Fontossági súly")
plt.show()

print(feature_importance_xgb.head(15))

#//////////////////////////////////////////////////////////////////////////////////////////////////
# POSZTONKÉNT MEGHATÁROZNI A LEGFONTOSABB STATISZTIKÁKAT

# Modell futtatása posztonként
def train_position_models(X, y):
    # Szétválasztani az adatokat tanuló és tesztelési halmazokra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adatok skálázása
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rand_for_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rand_for_model.fit(X_train_scaled, y_train)
    rand_for_importance = pd.Series(rand_for_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # XGBoost
    xgboost_model = xgb.XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
    xgboost_model.fit(X_train_scaled, y_train)
    xgboost_importance = pd.Series(xgboost_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return rand_for_importance, xgboost_importance

# Modell futtatása minden posztra
y_def = defenderPlayers["Rating"]
y_mid = midfielderPlayers["Rating"]
y_for = forwardPlayers["Rating"]

rand_for_def, xgb_def = train_position_models(X_def, y_def)
rand_for_mid, xgb_mid = train_position_models(X_mid, y_mid)
rand_for_for, xgb_for = train_position_models(X_for, y_for)

# Eredmények vizualizálása
def statistics_visualization(rf_importnace, xgb_importance, title):
    plt.figure(figsize = (15, 10))

    plt.subplot(1, 2, 1)
    sns.barplot(x = rf_importnace.head(10).index, y = rf_importnace.head(10).values, hue = rf_importnace.head(10).index, palette = "Blues")
    plt.xticks(rotation = 90)
    plt.title(f"Random Forest - {title}")

    plt.subplot(1, 2, 2)
    sns.barplot(x = xgb_importance.head(10).index, y = xgb_importance.head(10).values, hue = xgb_importance.head(10).index, palette = "Reds")
    plt.xticks(rotation = 90)
    plt.title(f"XGBoost - {title}")

    plt.tight_layout()
    plt.show()

# Védekező játékosok
statistics_visualization(rand_for_def, xgb_def, "Védők fontossági statisztikái")
print(rand_for_def)
print(xgb_def)

# Középpályások
statistics_visualization(rand_for_mid, xgb_mid, "Középpályások fontossági statisztikái")
print(rand_for_mid)
print(xgb_mid)

# Támadók
statistics_visualization(rand_for_for, xgb_for, "Támadók fontossági statisztikái")
print(rand_for_for)
print(xgb_for)

#//////////////////////////////////////////////////////////////////////////////////////////////////
# Posztra specifikus teljesítményelemző modellek felépítése

# Posztok szerint a legfontosabb statisztikai adatok kiválasztása
positions = {"D", "M", "F"}

# Modellépítő függvény - Random Forest és XGBoost
def train_and_evaulate_model(X, y, position_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators = 100, random_state = 42, eval_metric = 'rmse')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # Modell teljesítményének értékelése
    for name, pred in zip(["Random Forest", "XGBoost"], [rf_pred, xgb_pred]):
        print(f"\n[{position_name}] {name} Modell teljesítménye: ")
        print(f" MAE: {mean_absolute_error(y_test, pred):.4f}")
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        print(f" RMSE: {rmse:.4f}")
        print(f" R2: {r2_score(y_test, pred):.4f}")

    # Vizualizáció
    plt.figure(figsize = (15, 10))
    plt.scatter(y_test, pred, alpha = 0.5, label = name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linestyle = '--')
    plt.xlabel("Valódi értékek")
    plt.ylabel("Előrejelzett értékek")
    plt.title(f"{position_name} - {name} Előrejelzés vs Valós értékek")
    plt.legend()
    plt.show()

    return rf_model, xgb_model

# Lineáris regresszió
def train_and_evaluate_model_lin_regression(X, y, position_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    lin_reg_pred = lin_reg_model.predict(X_test)

    # Modell teljesítményének kiértékelése
    print(f"\n[{position_name}] Lineáris regresszió Modell teljesítménye: ")
    print(f" MAE: {mean_absolute_error(y_test, lin_reg_pred):.4f}")
    mse = mean_squared_error(y_test, lin_reg_pred)
    rmse = np.sqrt(mse)
    print(f" RMSE: {rmse:.4f}")
    print(f" R2: {r2_score(y_test, lin_reg_pred):.4f}")

    # Vizualizáció
    plt.figure(figsize = (15, 10))
    plt.scatter(y_test, lin_reg_pred, alpha = 0.5, label = "Lineáris regresszió")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linestyle = '--')
    plt.xlabel("Valódi értékek")
    plt.ylabel("Előrejelzett értékek")
    plt.title(f"{position_name} - Lineáris regresszió - Előrejelzés vs Valós értékek")
    plt.legend()
    plt.show()

    return lin_reg_pred

# Ridge, Lasso és ElasticNet regression modell tesztelése a védők szempontjából
def train_and_evaluate_model_ridge_lasso(X, y, position_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ridge regression
    ridge_model = Ridge(alpha = 1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)

    # Lasso regression
    lasso_model = Lasso(alpha = 0.1)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_pred = lasso_model.predict(X_test_scaled)

    # ElasticNet regression
    elastic_model = ElasticNet(alpha = 0.01, l1_ratio = 0.15)
    elastic_model.fit(X_train_scaled, y_train)
    elastic_pred = elastic_model.predict(X_test_scaled)

    # Modell teljesítményének a kiértékelése
    for name, pred in zip(["Ridge Regression", "Lasso Regression", "ElasticNet Regression"], [ridge_pred, lasso_pred, elastic_pred]):
        print(f"\n[{position_name}] {name} Modell teljesítménye: ")
        print(f" MAE: {mean_absolute_error(y_test, pred):.4f}")
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        print(f" MSE: {mean_squared_error(y_test, pred):.4f}")
        print(f" RMSE: {rmse:.4f}")
        print(f" R2: {r2_score(y_test, pred):.4f}")

    return ridge_model, lasso_model, elastic_model

# Modellek betanítása posztonként
def trains_models_per_position(data):
    for pos in positions:
        X_pos = data[data[f"Position_{pos}"] == 1].drop(columns = ["Rating"])
        y_pos = data[data[f"Position_{pos}"] == 1]["Rating"]

        if X_pos.shape[0] > 10: # Csak ha van elég minta
            train_and_evaulate_model(X_pos, y_pos, pos)
            train_and_evaluate_model_lin_regression(X_pos, y_pos, pos)
            train_and_evaluate_model_ridge_lasso(X_pos, y_pos, pos)

trains_models_per_position(statisticData)


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Adatok beolvasása az állományból
statisticData = pd.read_csv('C:/Users/Tamás/Desktop/Gazdinfó/Allamvizsga_masodikfelvonas/AllStatistics1.csv', encoding='ISO-8859-1')

# A "Minute Played" oszlopból eltávolítani a "'" karaktert, és átkonvertálni "int"-be 
statisticData['Minutes Played'] = statisticData['Minutes Played'].str.replace("'", "").astype('int64')

# Nem szükséges (nem fontos) adatokat tartalmazó oszlopok eltávolítása
statisticData.drop(columns = ["Team", "Player", "Notes", "Defence notes", "Pass notes"], inplace = True)

# A kategorikus változók numerikussá alakítása One-Hot Encoding-el
statisticData = pd.get_dummies(statisticData, columns = ["Position"], dtype = int)

# Csak numerikus típusú oszlopokat tartunk meg a feature setben
exclude_columns = ["Player", "MatchID", "Rating"]
feature_columns = [
    col for col in statisticData.columns
    if col not in exclude_columns and statisticData[col].dtype in [np.int64, np.float64, np.bool_]
]

X = statisticData[feature_columns]
y = statisticData["Rating"]

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

print(f"MAE_XGB: {mae_xgb: .3f}, R2_XGB: {r2_xgb: .3f}")

# A célváltozó meghatározása
X = statisticData.drop(columns = ["Rating"])
y = statisticData["Rating"]

# print(X.head())

# Szétválasztani az adatokat tanuló és tesztelési halmazokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# #//////////////////////////////////////////////////////////////////////////////////////////////////
#LINEÁRIS REGRESSZIÓ

# Lineáris regresszió modell betanítása
model = LinearRegression()
model.fit(X_train, y_train)

# Előrejelzés a teszthalmazon
y_pred = model.predict(X_test)

# Modell kiértékelése
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

#Előrejelzés scatter plot (y_test vs y_pred)
plt.figure(figsize = (8, 6))
plt.xlabel("Valós Rating értékek")
plt.ylabel("Előrejelzett Rating értékek")
plt.title("Lineáris regresszió: Valós vs. Előrejelzett értékek")
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = "red", linestyle = "--")
plt.show()

#Hibák eloszlása (y_test - y_pred)
plt.figure(figsize = (8, 6))
sns.histplot(y_test - y_pred, bins = 30, kde = True)
plt.xlabel("Hiba (Valós - Előrejelzett)")
plt.title("Lineáris regresszió: Hibák eloszlása")
plt.show()

#//////////////////////////////////////////////////////////////////////////////////////////////////
# #RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

#Random Forest regresszor létrehozása
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)

#Model betanítása
rf_model.fit(X_train, y_train)

#Előrejelzés a teszthalmazon
y_pred_rf = rf_model.predict(X_test)

#Kiértékelés
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MAE: {mae_rf:.4f}")
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}")
print(f"Random Forest R² Score: {r2_rf:.4f}")

#A Random Forest segítségével meg tudjuk nézni, hogy melyik változók a legfontosabbak a Rating előrejelzésében.

#Fontossági értékek kinyerése
features_importances = rf_model.feature_importances_

#Oszlopnevek hozzárendelése
features_names = X.columns

#Rendezés fontosság szerint
sorted_idx = np.argsort(features_importances)[::-1]

plt.figure(figsize = (12, 6))
sns.barplot(x = features_importances[sorted_idx][:10], y = [features_names[i] for i in sorted_idx[:10]])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Fontos Változó - Random Forest")
plt.show()

# #A Random Forest paramétereit tovább optimalizálhatjuk egy Grid Search segítségével.

from sklearn.model_selection import GridSearchCV

#Lehetséges paraméterek
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

#Grid Search futtatása
grid_search = GridSearchCV(RandomForestRegressor(random_state = 42), param_grid, cv = 3, scoring = "r2", n_jobs = 1)
grid_search.fit(X_train, y_train)

#Legjobb paraméterek kiírása
print(f"Best parameters: {grid_search.best_params_}")

#Legjobb modell kiértékelése
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)
r2_best_rf = r2_score(y_test, y_pred_best_rf)
print(f"Optimized Random Forest R² Score: {r2_best_rf:.4f}")

# Scatter plot a valós vs. előrejelzett értékekhez (Random Forest)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel("Valós Rating értékek")
plt.ylabel("Előrejelzett Rating értékek")
plt.title("Random Forest: Valós vs. Előrejelzett értékek")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.show()

# Hibák eloszlása (y_test - y_pred_rf)
plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred_rf, bins=30, kde=True)
plt.xlabel("Hiba (Valós - Előrejelzett)")
plt.title("Random Forest: Hibák eloszlása")
plt.show()

# Feature Importance vizualizáció
feature_importances = rf_model.feature_importances_
feature_names = X.columns

sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances[sorted_idx][:10], y=[feature_names[i] for i in sorted_idx[:10]])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Fontos Változó - Random Forest")
plt.show()

# Modell teljesítményének összehasonlítása
models = ["Linear Regression", "Random Forest"]
mae_values = [mae, mae_rf]
rmse_values = [rmse, rmse_rf]
r2_values = [r2, r2_rf]

plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color=['blue', 'green'])
plt.xlabel("Modellek")
plt.ylabel("MAE érték")
plt.title("Modell teljesítmény összehasonlítása - MAE")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['blue', 'green'])
plt.xlabel("Modellek")
plt.ylabel("RMSE érték")
plt.title("Modell teljesítmény összehasonlítása - RMSE")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, color=['blue', 'green'])
plt.xlabel("Modellek")
plt.ylabel("R² Score")
plt.title("Modell teljesítmény összehasonlítása - R² Score")
plt.ylim(0, 1)  # R² max értéke 1
plt.show()