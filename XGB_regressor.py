# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:31:23 2020

@author: Daniel.McGlynn
"""
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def XGB_regressor(normalised_x, normalised_y):
    """Implements XGBoost Regressor """
    data_dmatrix = xgb.DMatrix(data=normalised_x,label=normalised_y)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)
    
    xg_reg.fit(X_train,y_train)
    
    preds = xg_reg.predict(X_test)
    
    return preds

if __name__ == '__main__':
    XGB_regressor()