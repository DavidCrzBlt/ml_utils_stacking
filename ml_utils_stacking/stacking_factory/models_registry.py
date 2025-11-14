from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import xgboost as xgb

MODEL_CLASSES = {
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "HGB Regressor": HistGradientBoostingRegressor,
    "XGBoost": xgb.XGBRegressor,
    "SVR": SVR,
    "LightGBM": LGBMRegressor,
}

MODEL_MAPPER = {}
