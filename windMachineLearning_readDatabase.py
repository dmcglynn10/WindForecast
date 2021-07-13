# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:51:37 2018

@author: Daniel.McGlynn
Smoothes featues individually
Uses MinMaxScaler to Normalised data (0,1) and rescale to (-1,1)
Saves plots of training and test data to file

"""
import itertools as it
import msvcrt as m
import pandas as pd
import numpy as np
import math
import glob
import sys
import scipy
import pypyodbc as db
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import sklearn.metrics
from openpyxl import load_workbook
from sklearn.model_selection import cross_val_score
from pandas import ExcelWriter
from sklearn import ensemble
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn import neural_network
from sklearn.svm import SVR
import datetime as dt
from datetime import timedelta
from matplotlib.ticker import Formatter
import matplotlib.ticker as ticker
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import RFE
#from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
#import forestci as fci
import pickle
import neupy
#from neupy import algorithms, layers, estimators
import time
#ewma = pd.stats.moments.ewma
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf ,acf
from statsmodels.tsa.stattools import acf
from numpy import inf

from tkinter import filedialog
from tkinter import *

import saveAllGenResultsToExcel
import update_resultsTable


#data = pd.read_excel('dataframeSampleTemp.xlsx')
#data = pd.read_excel('DataframeDemandGasPrice.xlsx')
#data = pd.read_excel('DataframeSMP.xlsx')
def filter_holidays(w,hour, data):
    """This function filters holidays from data sample"""
    
    #list of holiday dates 
    hols = [dt.date(2016,12,27), dt.date(2016,12,26), dt.date(2016,8,29), dt.date(2016,5,30), dt.date(2016,5,2), 
            dt.date(2016,3,28), dt.date(2016,3,25), dt.date(2016,1,1),
            dt.date(2015,12,28), dt.date(2015,12,25), dt.date(2015,8,31), dt.date(2015,5,25), dt.date(2015,5,4), 
            dt.date(2015,4,6), dt.date(2015,4,3), dt.date(2015,1,1),
            dt.date(2018,5,28), dt.date(2018,5,7), dt.date(2018,4,2), dt.date(2018,3,30), dt.date(2018,1,1), 
            dt.date(2017,12,26), dt.date(2017,12,25),dt.date(2017,8,28), dt.date(2017,5,29), dt.date(2017,5,1),
            dt.date(2017,4,17)]
    
    #for i in range(0,len(hols)):
        #hols[i] = dt.datetime(hols[i].year, hols[i].month, hols[i].day, hour)
    if w == 0:
        data = data[(data['Weekday'] == 2) | (data['Weekday'] == 3) | (data['Weekday'] == 4)| (data['Weekday'] == 5) | (data['Weekday'] == 6) | (data['DateOnly'].isin(hols))]
    elif w == 1:
        data = data[(data['Weekday'] == 0) | (data['Weekday'] == 1) | (data['Weekday'] == 2) | (data['Weekday'] == 3) | (data['Weekday'] == 4)]
        data = data[(~data['DateOnly'].isin(hols))] 
    return data

def add_weekdayIndex(data):
    """This function adds a weekday index to the data"""
    #list of holiday dates 
    hols = [dt.datetime(2016,12,27), dt.datetime(2016,12,26), dt.datetime(2016,8,29), dt.datetime(2016,5,30), dt.datetime(2016,5,2), 
            dt.datetime(2016,3,28), dt.datetime(2016,3,25), dt.datetime(2016,1,1),
            dt.datetime(2015,12,28), dt.datetime(2015,12,25), dt.datetime(2015,8,31), dt.datetime(2015,5,25), dt.datetime(2015,5,4), 
            dt.datetime(2015,4,6), dt.datetime(2015,4,3), dt.datetime(2015,1,1),
            dt.datetime(2018,5,28), dt.datetime(2018,5,7), dt.datetime(2018,4,2), dt.datetime(2018,3,30), dt.datetime(2018,1,1), 
            dt.datetime(2017,12,26), dt.datetime(2017,12,25),dt.datetime(2017,8,28), dt.datetime(2017,5,29), dt.datetime(2017,5,1),
            dt.datetime(2017,4,17)]
    
    for i in range(0,data.shape[0]):
        if (data['Date'][i].weekday() >= 1) & (data['Date'][i].weekday() <= 4) & (~data['Date'].isin(hols)[i]):
            data.set_value(data.index[i], 'Weekday Index', 1)
        else:
            data.set_value(data.index[i], 'Weekday Index', 0)
    
    return data

def pick_most_correlated(data, hour):
    """This function picks the most correlated prices"""
    #autocorrelation
    #data.index = data['Date']
    data.reset_index(drop=True, inplace=True)
    
    data.dropna(inplace=True)
    
    data.reset_index(drop=True, inplace=True)
    
    #reverse order of data
    data = data.reindex(index=data.index[::-1])
    
    data.reset_index(drop=True, inplace=True)
    
    #discard all rows before the first row with data['Hour'] = hour
    data = data.iloc[ data[data['Hour']==hour].iloc[0].name:]
    
    data.reset_index(drop=True, inplace=True)
    
    #calculate acf
    acf, confint = statsmodels.tsa.stattools.acf(data['Price'], alpha=0.9, nlags=data.shape[0]-1)
    #plot acf
    fig, ax = plt.subplots(figsize=(10,10))
    #fig = plot_acf(data['Price'], alpha=0.01, use_vlines=False, marker='.', ax=ax, lags=data.shape[0]-1)
    #print(fig)
    plt.xlim(xmin = 0, xmax=100)
    plt.ylim(ymin = -0.2, ymax=1.2) 
    plt.title('Acf all data')
    plt.legend('acf')
    plt.grid()
    #plt.savefig(file_path_test+'/'+'results_'+hours_str+'_acf_allData_plot_BETTA_99cf_1'+days_str+'.png', bbox_inches='tight')

    
    l = find_lag_prices(acf, confint, data)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    #fig = plot_acf(data['Price'], alpha=0.01, use_vlines=False, marker='.', ax=ax, lags=data.shape[0]-1) 

    #print(fig)
    
    #data = data.reindex(index=data.index[::-1])
    
    #data1 = data[(data['Hour'] == hour)]
    
    data = data.ix[l]
    
    #frames = [data1,data2]
    
    #data = pd.concat(frames)
    
    #data = data.drop_duplicates(subset=['Date'])
    
    data.sort_index(inplace=True)
    
    #data = data.reindex(index=data.index[::-1])
    
    data = data.reset_index(drop=True)
    
    data = data.reindex(index=data.index[::-1])
    
    return data

def find_lag_prices(cf, confint, data ):
    #This function finds indices of lag prices that are outside the confidence limits.
    a = np.zeros(confint.shape)
    
    for i in range(0,confint.shape[0]):
        a[i,0] = confint[i,0] - cf[i]
        a[i,1] = confint[i,1] - cf[i]
    
    p = np.zeros(cf.shape)
    
    for i in range(0,cf.shape[0]):
        if (((abs(cf[i]) > abs(a[i,0])) & (abs(cf[i]) > abs(a[i,1])) & (abs(cf[i]) > 0.0)) | (data['Hour'].iloc[i] == hour)):# | (data['Hour'].iloc[i] == hour+12)):
            p[i] = i
    
    X = np.ma.masked_equal(p,0)       
    X = X.compressed()
    X = [int(i) for i in X]
    
    return X



def filter_data(data, hours_str, days_str, feature_list, no_days):
    
    #data = data[feature_list]
    
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'Hour', data['Date'].iloc[i].hour) #create weekday column
        
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'Month', data['Date'].iloc[i].month) #create month column
    
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'Weekday', data['Date'].iloc[i].weekday()) #create weekday column
        
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'DateOnly', data['Date'].iloc[i].date()) #create date column
        
    
    #data = data[data['DateOnly'].isin(data[data['Price'] > 10]['DateOnly'].unique())]


    data = data[(((data['Date'] >= start_date) & (data['Date'] <= end_date)))]# |
    
    """
    data = data[(data['Forecast_Wind_Generation_all_mean'] <= 2500) & (data['Forecast_Wind_Generation_all_mean'] >= 1500)
    & (data['Actual_System_Demand_all_mean'] >= 3500) & (data['Actual_System_Demand_all_mean'] <= 4500) & (data['Price_Mean'] >= 35) & (data['Price_Mean'] <= 45)]
    """
    #        ((data['Date'] >= dt.date(2016,1,1)) & (data['Date'] <= dt.date(2016,1,1))))]
    
    #data = pick_most_correlated(data, hour)
    
    #Select peak or off-peak
    if p == 0:
        #data = data[(data['Hour'] == hour)]# & (data['Hour'] <= 16)] #filer hours (off peak)
        h = 2
        #data = data[(data['Hour'] >= (dt.datetime(2019,1,1,hour)-timedelta(hours=h)).hour) | (data['Hour'] <= (dt.datetime(2019,1,1,hour)+timedelta(hours=h)).hour)]
        """
        if hour == 22:
            data = data[(data['Hour'] >= 22) | (data['Hour'] == 0) | (data['Hour'] >= hour-h)]
        if hour == 23:
            data = data[(data['Hour'] >= hour-h) | (data['Hour'] == hour) | (data['Hour'] <= 1) ] #filer hours equal to x
        elif hour == 0:
             data = data[(data['Hour'] >= 24-h) | (data['Hour'] == hour) | (data['Hour'] <= hour+h)]
        elif hour == 1:
             data = data[(data['Hour'] >= 24-h) | (data['Hour'] == hour) | (data['Hour'] <= hour+h)]
        elif hour == 2:
             data = data[(data['Hour'] >= 0) | (data['Hour'] == hour) | (data['Hour'] <= hour+h)]
        else :
            data = data[(data['Hour'] >= hour-h) | (data['Hour'] == hour) | (data['Hour'] <= hour+h)]
        #hours_str = 'off-peak'
        """
    elif p == 1:
        data = data[(data['Hour'] > 14) & (data['Hour'] < 2)] #filer hours (on peak)
        hours_str = 'peak'
      
    #Select weekend or weekdays
    if w == 0:
        days_str = 'weekend'
    elif w == 1:
        days_str = 'weekdays'
    elif w == 2:
        days_str = 'all_days'
    
    if (w == 0) | (w == 1):
        data  = filter_holidays(w,hour, data)
    
    data = data[feature_list]
    
    data = data.iloc[:,:]

    data_dates = data['Date'][:]
    
    return data, hours_str, days_str, data_dates
#months
#data = data[(data['Month'] == 1) | (data['Month'] == 12) | (data['Month'] == 11)]


#data['Prices Avg'] = (data['Previous Day Price'])
def filter_99pc(data, hour, acf):
    "This function filters prices between start date and end date that are 99percent autocorrelated"
    df_columns = data.columns
    
    start_datetime = dt.datetime(2017,1,1,hour)
    end_datetime = dt.datetime(2018,1,1)
    
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'DateOnly', data['Date'][data.index[i]].date())
        
    df_before = data[(data['Date'] < start_datetime)]
    
    df_after = data[(data['Date'] >= end_datetime)]
        
    df = data[(((data['Date'] >= start_datetime) & (data['Date'] < end_datetime)))]
    
    df = df[df_columns]
    df_before = df_before[df_columns]
    df_after = df_after[df_columns]
    
    #caluculate acf
    acf, confint = acf(df['Price'], alpha=0.01, nlags=df.shape[0]-1)
    
    #This function finds indices of lag prices that are outside the confidence limits.
    a = np.zeros(confint.shape)
    
    for i in range(0,confint.shape[0]):
        a[i,0] = confint[i,0] - acf[i]
        a[i,1] = confint[i,1] - acf[i]
    
    p = np.zeros(acf.shape)
    
    for i in range(0,acf.shape[0]):
        if ((acf[i] < a[i,0]) | (acf[i] > a[i,1])) & (math.fabs(acf[i]) > 0.0):
        #if (math.fabs(acf[i]) > 0.5):
            p[i] = i + 1
    
    X = np.ma.masked_equal(p,0)       
    X = X.compressed()
    X = [int(i) for i in X]
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    df1 = df.iloc[X]
    df2 = df.iloc[X[len(X)-1]:]
    frames  = [df_before, df1, df2, df_after]
    data = pd.concat(frames)
    data = data.reset_index(drop=True)
    return data

def get_lag_prices(data, feature_list, acf, mode):
    """Adds lag prices to dataframe"""
    
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
    elif(regressor_select == 6):
        model_str = 'GRNN'
    
    date_list = ['Date', 'Price']
    
    lag_vars = []
    
    for i in range(0,len(feature_list)):
        lag_vars.append(feature_list[i])
    
    for i in range(0,len(date_list)):
        lag_vars.remove(date_list[i])
        
    
    
    """
                'Actual_Wind_Generation_all', 'Actual_System_Demand_all',
                'Interconnector_NET', 'BETTA_Price',
                'total_unit_availability',
                'imbalance_price', 'total_pn',
                'total_unit_availability',
                'pmea', 'qpar', 'administered_scarcity_price',
                'market_backup_price', 'short_term_reserve_quantity',
                'operating_reserve_requirement',
                'tso_demand_forecast', 'tso_renewable_forecast', 'default_price_usage',
                'asp_price_usage']
    """
    """
    remove_vars = ['Price_lag', 'Actual_Wind_Generation_all', 'Actual_System_Demand_all', 'temperature']
    
    for i in range(0,len(remove_vars)):
        lag_vars.remove(remove_vars[i])
    
    #add lag prices to dataframe
    for var in remove_vars:
        for i in range(24,1920):
            if (i % 6 == 0):
                data[var+' -'+str(i)] = data[var].shift(i)
                
    for var in remove_vars:
        for i in range(24,1920-24):
            if (i % 6 == 0):
                data[var+'_Diff -'+str(i)] = data[var+' -'+str(i)] - data[var+' -'+str(i+6)]
    """
    
    #add lag prices to dataframe
    for var in lag_vars:
        if var in feature_list:
            if var in ['Price_lag']:
                for i in range(24,36):
                    if ((i % 24 == 0) | (i % 24 == 23) | (i % 24 == 1) | (i % 24 == 2) | (i % 24 == 22) & (i >= 24)):
                        data[var+' -'+str(i)] = data[var].shift(i)
            elif var[-5:] in ['_vols', '_Bids', 'Gen_total_volume_vols', 'Sup_total_volume_vols']:
                for i in range(24,36):
                    if ((i % 24 == 0) | (i % 24 == 23) | (i % 24 == 1) | (i % 24 == 2) | (i % 24 == 22) & (i >= 24)):
                        data[var+' -'+str(i)] = data[var].shift(i)          
            elif var in ['load_forecast_ni', 'load_forecast_roi', 'Forecast_Wind_Generation_all',
                         'GU_400762', 'GU_400930','GU_400480', 'GU_400500', 'GU_400530', 'GU_400540',
                        'GU_500040', 'GU_400850','DSU_401610', 'DSU_401330']:
                 for i in range(0,36):
                    if ((i % 24 == 0) | (i % 24 == 23) | (i % 24 == 1) | (i % 24 == 2) | (i % 24 == 22) | (((i>0) & (i<10)))):
                        data[var+' -'+str(i)] = data[var].shift(i)
                
    
    """
    #add lag differences to the dataframe
    for var in lag_vars:
        for i in range(24,672-24):
            if (i % 1 == 0):
                data[var+'_Diff -'+str(i)] = data[var+' -'+str(i)] - data[var+' -'+str(i)]
                #if var in ['Price_lag', 'Gas_Price', 'imbalance_price', 'total_pn']:
                    #data[var+'rolling -'+str(i)] = data[var+' -'+str(i)].rolling(window=24).mean()
    """

    
    #remove nan from dataframe    
    data = data[np.isfinite(data[data.columns[-1]])]
    
    data_columns = data.columns
    
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'DateOnly', data['Date'].loc[data.index[i]].date()) #create date column
    
    data = data[(((data['Date'] >= start_date_tt) & (data['Date'] <= end_date_tt)))]
    
    data = data[data_columns]
    
    #calculate acf
    acf, confint = acf(data['Price'], alpha=0.99, nlags=data.shape[0]-1)
    #plot acf
    fig, ax = plt.subplots(figsize=(10,10))
    #fig = plot_acf(data['Price'], alpha=0.99, use_vlines=False, marker='.', ax=ax, lags=data.shape[0]-1)
    #print(fig)
    #plt.xlim(xmin = 0, xmax=120)
    #plt.ylim(ymin = -0.2, ymax=1.2) 
    plt.title('Acf filtered data')
    plt.legend('acf')
    plt.grid()
    #plt.savefig(file_path_test+'/'+'results_'+hours_str+'_acf_filteredData_plot_BETTA_99cf_'+days_str+'.png', bbox_inches='tight')
    
    #This function finds indices of lag prices that are outside the confidence limits.
    a = np.zeros(confint.shape)
    
    for i in range(0,confint.shape[0]):
        a[i,0] = confint[i,0] - acf[i]
        a[i,1] = confint[i,1] - acf[i]
    
    p = np.zeros(acf.shape)
    
    if mode == 'training':
        for i in range(0,len(acf)):
            if (((acf[i] < a[i,0]) | (acf[i] > a[i,1])) & (math.fabs(acf[i]) > 0.0)):
            #if (math.fabs(acf[i]) > 0.5):
                p[i] = i + 1
    elif mode == 'Predict':
         for i in range(0,len(acf)):
            #if (math.fabs(acf[i]) > 0.5):
            p[i] = i + 1
    
    X = np.ma.masked_equal(p,0)       
    X = X.compressed()
    X = [int(i) for i in X]
    
    #put lag price indices into a list of strings "Price -1" ...
    l = []
    for var in lag_vars:
        if var in feature_list:
            for i in range(0,len(X)):
                l.append(var+" -"+str(X[i]))
                l.append(var+'_Diff -'+str(X[i]))
    #print(l)
    

    l = list(set(data.columns) & set(l)) #use common elements in data and l

    #Lag prices from parital autocorrelation
    
    """
    df = data[["Price -1", "Price -2", "Price -3", "Price -4", "Price -20", "Price -21", "Price -22", "Price -23", "Price -24", "Price -25",
                 "Price -26", "Price -27", "Price -29", "Price -46", "Price -47", "Price -48", "Price -49", "Price -69", "Price -70", "Price -71", 
                 "Price -72", "Price -73", "Price -92", 
                 "Price -93", "Price -94", "Price -95", "Price -96", "Price -97", "Price -99", "Price -117", "Price -119", "Price -120", "Price -121", 
                 "Price -122", "Price -123", "Price -124", "Price -141", "Price -142", "Price -143", "Price -144", "Price -145", "Price -146", "Price -165", 
                 "Price -166", "Price -167", "Price -168", "Price -169", "Price -188", "Price -189", "Price -190", "Price -191", "Price -193", "Price -215", 
                 "Price -219", "Price -220", "Price -239", "Price -241"]]
                 #, "Price -261", "Price -265", "Price -266", "Price -285", "Price -286", "Price -310", 
                 #"Price -313", "Price -333", "Price -334", "Price -335", "Price -336", "Price -337", "Price -338", "Price -333", "Price -334", "Price -335", 
                 #"Price -336", "Price -337", "Price -338",  "Price -357", "Price -358", "Price -361",  "Price -362", "Price -406", "Price -408",
                 #"Price -409", "Price -429", "Price -430", "Price -434", "Price -435", "Price -457", "Price -458", "Price -479"]]

    #Lag prices from autocorrelation
    """
    
    df = data[l]
    """
    df = data[["Price -1", "Price -2", "Price -3", "Price -23", "Price -24", "Price -25", 
             "Price -47", "Price -48", "Price -49", "Price -71", "Price -72", "Price -73", "Price -95", 
             "Price -96", "Price -97", "Price -119", "Price -120", "Price -121", "Price -143", "Price -144",
             "Price -145", "Price -167", "Price -168", "Price -169", "Price -191", "Price -192", "Price -193",
             "Price -215", "Price -216", "Price -217", "Price -239", "Price -240", "Price -241"]]#, "Price -34", 
             #"Price -35", "Price -36", "Price -37", "Price -38", "Price -39", "Price -40", "Price -41", 
             #"Price -42", "Price -43", "Price -44", "Price -45", "Price -46", "Price -47", "Price -48"]]
            # "Price -49", "Price -50", "Price -51", "Price -52", "Price -53", "Price -54", "Price -55", 
             #"Price -56", "Price -57", "Price -58", "Price -59", "Price -60", "Price -61", "Price -62", 
             #"Price -63", "Price -64", "Price -65", "Price -66", "Price -67", "Price -68", "Price -69", 
             #"Price -70", "Price -71", "Price -72", "Price -73", "Price -74", "Price -75", "Price -76", 
             #"Price -77"]]
    
    df1 = pd.DataFrame()
    
    df1 = data.iloc[:,150:]
    
    
    df = df.dropna()
    
    a = []
    lag_price_labels = []
    
    cov_data = np.corrcoef(df.T)
    
    for i in range(1,len(cov_data[0])):
        if np.abs(cov_data[0][i]) > 0.000000005:
            a.append(i)
            
    for i in a:
        lag_price_labels.append(df.columns[i])
        #lag_price_labels.append(df1.columns[i])
        #lag_price_labels.append(df1.columns[i+119])
        #lag_price_labels.append(df1.columns[i+2*119])
        #lag_price_labels.append(df1.columns[i+3*119])
    
    """
    for i in df.columns:
        feature_list.append(i)
        
    feature_list = data.columns.tolist()
        
    #print(feature_list)
        
    for i in range(0,data.shape[0]):
        data.set_value(data.index[i], 'Hour', data['Date'].iloc[i].hour) #create Hour column
        
    #filter data by hour
    #data = data[(data['Hour'] == hour)]
    """
    try:
        for i in range(0,len(lag_vars)):
            feature_list.remove(lag_vars[i])
    except:
        print('Price_lag not in feature_list')
    """    
        
    data = data[feature_list]
    
    
    return lag_price_labels, feature_list, data



#data['TSD'] = data['TSD']**1.25
#data['Price'] = data['Price']**-0.5
#data['Gas'] = data['Gas']**0.9
#data['Wind1']  = data['Wind1']**0.25
#data['Solar'] = data['Solar']

#data.replace(0,np.nan, inplace=True)

def createDirectory(start_date_tt, end_date_tt, regressor_select, no_days, folder_path):
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
    elif(regressor_select == 6):
        model_str = 'GRNN'
    
    file_path_train = folder_path+model_str+'/WeekendWeekdays'+start_date_tt.strftime('%Y-%m-%d')+'_'+end_date_tt.strftime('%Y-%m-%d')+'/Train'
    file_path_test = folder_path+model_str+'/WeekendWeekdays'+start_date_tt.strftime('%Y-%m-%d')+'_'+end_date_tt.strftime('%Y-%m-%d')+'/Test'
    
    
    if not os.path.exists(file_path_train):
        os.makedirs(file_path_train)
        
    if not os.path.exists(file_path_test):
        os.makedirs(file_path_test)
    
    return file_path_train, file_path_test

def smoothvars(data, data_dates, g, feature_list):
    "This function smooths model variables"
    ##### choose which features to smooth ######
    window_sizes = [1] * len(feature_list)
    for i in range(1,len(feature_list)):
        #if i in [3,4]:
        window_sizes[i] = 25#max(np.std(data[feature_list[i]])*2,1)c

    
    data = data.set_index('Date')
    for i in range(2, data.shape[1]):
        data[data.columns[i]] = data[data.columns[i]].rolling(window=math.ceil(window_sizes[i])).mean()
    #data = data[np.isfinite(data[data.columns[np.argmax(window_sizes)]])]
    data = data[np.isfinite(data[data.columns[2]])]
    data_dates = data.index
    """
    g = gaussian_filter1d(data['Price'], sigma=2)
    g_prevWeek = gaussian_filter1d(feat_vars['Previous Week Price'], sigma=2)
    g_prevHour = gaussian_filter1d(feat_vars['Previous Hour Price'], sigma=2)
    g_Price = gaussian_filter1d(feat_vars['Price'], sigma=2)
    
    feat_vars['Previous Week Price Sm'] = g_prevWeek
    feat_vars['Previous Hour Price Sm'] = g_prevHour
    feat_vars['Price Sm'] = g_Price
    """
    print(len(g))
    return data, data_dates

def smoothvars_exponential(data, data_dates, g, feature_list, features_to_smooth):
    "This function smooths model variables using exponential smoothing"
    ##### choose which features to smooth ######
   
    for i in range(1,len(feature_list)):
         if feature_list[i] in features_to_smooth:
             # take EWMA in both directions with a smaller span term
             fwd = ewma( data[feature_list[i]], span=25) # take EWMA in fwd direction
             bwd = ewma( data[feature_list[i]][::-1], span=25) # take EWMA in bwd direction
             c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
             c = np.mean( c, axis=0 ) # average
             data[feature_list[i]] = c
    
    data = data.set_index('Date')
    
    #data = data[np.isfinite(data[data.columns[np.argmax(window_sizes)]])]
    data_dates = data.index
    
    print(len(g))
    return data, data_dates

def smoothvars_test(normalised_x, feature_list, features_to_smooth):
    "This function smooths model variables using exponential smoothing"
    ##### choose which features to smooth ######
    max_new = 100
    min_new = 0
   
    for i in range(1,len(feature_list)):
         if feature_list[i] in features_to_smooth:
             
             max_old = max(normalised_x[feature_list[i]])
             min_old = min(normalised_x[feature_list[i]])
              
             spn = ((max_new - min_new) / (max_old - min_old)) * (int(np.std((normalised_x[feature_list[i]]))) - max_old) + max_new
             # take EWMA in both directions with a smaller span term
             fwd = ewma( normalised_x[feature_list[i]], span=spn) # take EWMA in fwd direction
             bwd = ewma( normalised_x[feature_list[i]][::-1], span=spn) # take EWMA in bwd direction
             c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
             c = np.mean( c, axis=0 ) # average
             normalised_x[feature_list[i]] = c
    
    print(len(g))
    return normalised_x

def normalisevars(data):
    "This function normalises variables"
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    normalised = scaler.transform(data)
    inversed = scaler.inverse_transform(normalised)
    print(len(normalised))
    #scaler = MinMaxScaler(feature_range=(0,1))
    #scaler = scaler.fit(feat_vars)
    #normalised = scaler.transform(feat_vars)
    #inversed = scaler.inverse_transform(normalised)
    return normalised

def rescaleData(data, x1, x2, y, UL, days_str, hours_str, model_str, mode):
    "Uses MinMaxScaler to normalised or rescale dataset"
    x1 = data.iloc[:, 1]
    x2 = data.iloc[:,1:]
    y = data.iloc[:,0]
    print(x1.shape)
    
    """
    for i in range(0,y.shape[0]):
        if y[i] > UL:
            y[i] = UL + UL*math.log10( y[i] / UL)
    """
            
    if mode == 'training': #training
        #min_max_scaler_x = preprocessing.MinMaxScaler(feature_range=(0,1)) #normalised to (0,1)
        #min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(0,1)) #normalised to (0,1)
        #min_max_scaler_x1 = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
        min_max_scaler_x2 = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
        min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
        #min_max_scaler_x2 = preprocessing.StandardScaler() #standard scaler
        #min_max_scaler_y = preprocessing.StandardScaler() #standard scaler
           
        #min_max_scaler_x1 = min_max_scaler_x1.fit(x1)
        min_max_scaler_x2 = min_max_scaler_x2.fit(x2)
        min_max_scaler_y = min_max_scaler_y.fit(y.values.reshape(len(y),1))
        #normalised_x[:,0] = min_max_scaler_x1.transform(x1)
        
        # save the min_max_scaler model to disk
        filename_x = 'min_max_scaler_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_x.sav'
        filename_y = 'min_max_scaler_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_y.sav'
        with open(folder_saveModels+'\\'+filename_x,'wb') as f_x:
            pickle.dump(min_max_scaler_x2, f_x)
        with open(folder_saveModels+'\\'+filename_y,'wb') as f_y:
            pickle.dump(min_max_scaler_y, f_y)
            
    elif mode == 'Predict': #testing   
        # load the min_max_scaler model from disk
        min_max_scaler_x2 = pickle.load(open(folder_saveModels+'\\'+'min_max_scaler_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_x.sav', 'rb'))
        min_max_scaler_y = pickle.load(open(folder_saveModels+'\\'+'min_max_scaler_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_y.sav', 'rb'))
       
    normalised_x = min_max_scaler_x2.transform(x2)#x2
    normalised_y = min_max_scaler_y.transform(y.values.reshape(len(y),1))#y
    
    normalised_x = pd.DataFrame(data = normalised_x, index = data.index, columns = data.columns[1:])
    normalised_y = pd.DataFrame(data = normalised_y, index = data.index, columns = data.columns[0:1])
    
    #normalised_x = pd.DataFrame(data=normalised_x[:,:], columns=data.columns[1:], index=data.index)
    #normalised_y = pd.DataFrame(data=normalised_y[:,0], columns=data.columns[0:1], index=data.index)
    return normalised_x, normalised_y, min_max_scaler_y, min_max_scaler_x2

def findoutliers(normalised_x, normalised_y):
    "This function detects oultlier"
    x1 = data.iloc[:, 1]
    x2 = data.iloc[:,1:]
    y = data.iloc[:,0]
    print(x1.shape)
    #min_max_scaler_x = preprocessing.MinMaxScaler(feature_range=(0,1)) #normalised to (0,1)
    #min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(0,1)) #normalised to (0,1)
    #min_max_scaler_x1 = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
    #min_max_scaler_x2 = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
    #min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(-1,1)) #rescaled to (-1,1)
    min_max_scaler_x2 = preprocessing.StandardScaler() #standard scaler
    min_max_scaler_y = preprocessing.StandardScaler() #standard scaler
       
    #min_max_scaler_x1 = min_max_scaler_x1.fit(x1)
    min_max_scaler_x2 = min_max_scaler_x2.fit(x2)
    min_max_scaler_y = min_max_scaler_y.fit(y.values.reshape(len(y),1))
    #normalised_x[:,0] = min_max_scaler_x1.transform(x1)
    
    x_1 = min_max_scaler_x2.transform(x2)
    y_1 = min_max_scaler_y.transform(y.values.reshape(len(y),1))
    
    np.std(data['Price'], axis=0)
    outlier_rows= np.where(np.abs(y_1)>2)
    return outlier_rows

def findoutliers_LocalOutlierFactor(normalised_x, normalised_y):
    "This function detects outliers"
    np.random.seed(42)
    clf = LocalOutlierFactor(n_neighbors = 15)
    
    y_pred = clf.fit_predict(data['Price'].reshape(len(data),1))
    outlier_rows = y_pred
    return outlier_rows

def clipoutliers(normalised_x, normalised_y):
    "This function clips outliers"
    print(len(normalised_x))
    for i in outlier_rows:
        for j in outlier_columns:
            normalised_x[i,j] = (normalised_x[i,j] / np.abs(normalised_x[i,j])) * 2
    return

def set_sample_weights(normalised_x, outlier_rows):
    "This functions sets sample weights"
    weights = np.ones(len(normalised_x))
    for i in outlier_rows:
        weights[i] *= 5
    return weights

def set_sample_weights_LOF(normalised_x, outlier_rows):
    "This functions sets sample weights"
    weights = np.ones(len(normalised_x))
    for i in range(0,len(outlier_rows)):
        if outlier_rows[i] == -1:
            weights[i] *= 15
    return weights
    

def clipvars():
    "Clips variables"
    up_lim_Price = np.mean(data['Price']) + 2 * np.std(data['Price'])

    lo_lim_Price = np.mean(data['Price']) - 2 * np.std(data['Price'])

    up_lim_PrevWeek = np.mean(data['Previous Week Price']) + 2 * np.std(data['Previous Week Price'])

    lo_lim_PrevWeek = np.mean(data['Previous Week Price']) - 2 * np.std(data['Previous Week Price'])

    up_lim_PrevHour = np.mean(data['Previous Hour Price']) + 1.5 * np.std(data['Previous Hour Price'])

    lo_lim_PrevHour = np.mean(data['Previous Hour Price']) - 1.5 * np.std(data['Previous Hour Price'])
    
    a_clipped = np.clip(a, a_min = lo_lim, a_max = up_lim)
    a_prevWeek_clipped = np.clip(a, a_min = lo_lim_PrevWeek, a_max = up_lim_PrevWeek)
    a_prevHour_clipped = np.clip(a, a_min = lo_lim_PrevHour, a_max = up_lim_PrevHour)
    return

def tunevars():
    "This function tunes variables using RandomizedGridCV"
    
    return

def definearrays(normalised_x, normalised_y, X_train, y_train, X_test, y_test, t_size, data, feature_list, w):
    "define arrays that will hold features and target variables"
    X = data[feature_list[2:]]
    y = data['Price']
    
    icount = 0
    
    features = feature_list
    l = [features[2:]]
    
    MAE_arr = np.empty([len(l)])
    root_MSE_arr = np.empty([len(l)])
    R2_arr = np.empty([len(l)])
    
    print(len(normalised_x))
    
    #Split set into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(
       #normalised[:,2:], data['Price'][data.shape[0]-len(normalised):], test_size=0.4, random_state=0)
    
    #Split set into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
       normalised_x, normalised_y, test_size=no_days, shuffle=False)
        
    predictions_arr = np.empty([len(y_test),len(l)])
    predictions_train = np.empty([len(y_train),len(l)])
    predictions_train_best = np.empty([len(y_train),len(l)])
    price_V_unbiased = np.empty([len(y_test), len(l)])
    
    return X_train, y_train, X_test, y_test, X, y, l, predictions_arr, root_MSE_arr, predictions_train, predictions_train_best, price_V_unbiased

def tune_RF():
    "This function tunes Random forest hyperparameters"
    lm = ensemble.RandomForestRegressor()
    
    
    return 

def traintestalgorithm(X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, w, elapsed_times, UL, foldername, nn_layer_size, folder_saveModels):
    "This function trains and tests a regression algorithm"
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
        
        
    RMSE = 100000
    R2_best  = 0
    num_iter = 0
    v = 0
    icount = 0
    
    predictions_arr = np.empty([len(y_test),len(l)])
    predictions_train = np.empty([len(y_train),len(l)])
    predictions_train_best = np.empty([len(y_train),len(l)])
    price_V_unbiased = np.empty([len(y_test), len(l)])
    
    #arrays to hold training and test times
    times = np.zeros(2)
        
    while (RMSE > 1): 
        
        #Tune Random Forest Regressor
        """
        lm = ensemble.RandomForestRegressor()
        
        n_estimators_RF = np.arange(2,10,1)
        min_samples_leaf_RF = np.arange(2,30,1)
        min_impurity_decrease_RF = np.arange(0.0, 0.2, 0.05)
        max_leaf_nodes_RF = np.arange(10,50,1)
        max_depth_RF = np.arange(40,120,10)
        random_state_RF = np.array([70])
        
        param_distributions_RF = dict(n_estimators = n_estimators_RF, min_samples_leaf = min_samples_leaf_RF,
                                          min_impurity_decrease = min_impurity_decrease_RF, 
                                          max_leaf_nodes = max_leaf_nodes_RF,
                                          max_depth = max_depth_RF,
                                          random_state = random_state_RF)
        
        grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_RF, n_iter=10)
        
        grid.fit(X_train, y_train)
    
        grid.best_params_
        
        lm = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], min_samples_leaf = grid.best_params_['min_samples_leaf'],
                                   min_impurity_decrease = grid.best_params_['min_impurity_decrease'], max_leaf_nodes = grid.best_params_['max_leaf_nodes'],
                                   max_depth = grid.best_params_['max_depth'], random_state = grid.best_params_['random_state'])
        
        n_estimators_RF_tuned = grid.best_params_['n_estimators'] 
        min_samples_leaf_RF_tuned = grid.best_params_['min_samples_leaf']
        min_impurity_decrease_RF_tuned = grid.best_params_['min_impurity_decrease']
        random_state_RF_tuned = grid.best_params_['random_state']
        max_leaf_nodes_RF_tuned = grid.best_params_['max_leaf_nodes']
        max_depth_RF_tuned = grid.best_params_['max_depth']
        """
        
        #lm = SVR()
        #lm = linear_model.Lasso(alpha = 0.001)
        #lm = linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.5)
        #lm = linear_model.HuberRegressor(alpha = 0.01, epsilon = 2)
        #lm = linear_model.Ridge(alpha = 0.01)
        #lm = linear_model.LinearRegression()
        #lm = tree.DecisionTreeRegressor()
        if(regressor_select == 0):
            lm = ensemble.GradientBoostingRegressor()
            
        if(regressor_select == 4):
            lm = ensemble.BaggingRegressor()
        #lm = ensemble.RandomForestRegressor(n_estimators = 2, max_features=3, max_depth = 10, min_samples_split = 4,
        #                                    min_samples_leaf = 5, min_weight_fraction_leaf = 0.01, max_leaf_nodes = 5, 
        #                                    min_impurity_split = 0.5)
        #
        if(regressor_select == 1):
            lm = ensemble.AdaBoostRegressor()
            
        if(regressor_select == 2):    
            lm = neural_network.MLPRegressor()
        
        if(regressor_select == 3):
            lm = SVR()
        if(regressor_select == 5):
            lm = ensemble.RandomForestRegressor()
        
        #grid = RandomizedSearchCV(estimator=lm, param_distributions=dict(C=C, epsilon=epsilon, degree=degree, gamma=gamma), n_iter=30)
        #grid = RandomizedSearchCV(estimator=lm, param_distributions=dict(C=C, epsilon=epsilon, degree=degree, gamma=gamma), n_iter=30)
        #grid = RandomizedSearchCV(estimator=lm, param_distributions=dict(alpha=alpha), n_iter=30)
        n_estimators = np.arange(300, 600, 50)
        max_depth = np.arange(2,10,1)
        min_samples_split = np.arange(2,6,1)
        learning_rate = np.arange(0.01, 1, 0.01)
        max_samples = np.arange(1,8,1)
        max_features = np.arange(1,8,1)
        loss = ['ls','lad', 'huber' , 'quantile']
       
        #Ada Boost 
        if(regressor_select == 1):
            base_estimator_AdaBoost = [ExtraTreesRegressor(n_estimators = 100,
                                                           random_state = 70,
                                                           min_samples_leaf=26,
                                                           max_depth = 300,
                                                           max_leaf_nodes = 20,
                                                           min_impurity_decrease = 0.0)]
            loss_AdaBoost = ['linear', 'square', 'exponential']
            learning_rate_AdaBoost = np.arange(0.01, 5, 0.001)
            n_estimator_AdaBoost = np.arange(10, 11, 1)
            random_state_AdaBoost = np.array([70])
        #Gradient Boost
        if(regressor_select == 0):
            loss_GradBoost = ['ls','lad', 'huber' , 'quantile']
            learning_rate_GradBoost = np.arange(0.5, 0.6, 0.1) #flaot
            n_estimators_GradBoost = np.arange(90, 110, 1) #int (default 100)
            max_depth_GradBoost = np.arange(2,3,1) #int
            criterion_GradBoost = ['mae']#['friedman_mse', 'mse', 'mae'] 
            min_samples_split_GradBoost = np.arange(2,3,1) #int
            min_samples_leaf_GradBoost = np.arange(1,5,1) #int
            subsample_GradBoost = np.arange(0.8,1,0.1) #must be between (0,1)
            max_features_GradBoost = [None]#['auto', 'sqrt', 'log2', None] #int
            max_leaf_nodes_GradBoost = np.arange(2,5,1) #int
            min_impurity_split_GradBoost = np.arange(0.55,0.6,0.01) #float
            alpha_GradBoost = np.arange(0.5,0.6,0.1) #float
            #init = np.arange()
            verbose_GradBoost = np.arange(1,2,1) #int
            warm_start_GradBoost = np.array([True]) #bool
            random_state_GradBoost = np.arange(6,7,1) #int
            presort_GradBoost = np.array([True]) #bool
        print(y_test.index)

        #MLP
        if(regressor_select == 2):
            if (len(data.columns) > 11):
                hidden_layer_sizes_MLP =  nn_layer_size#(len(data.columns)-math.ceil(len(data.columns)/4), len(data.columns) - math.ceil(len(data.columns)/2))
                print(str(hidden_layer_sizes_MLP))
                #print(feature_list)
                print(len(feature_list))
            else:
                hidden_layer_sizes_MLP =  nn_layer_size#(len(data.columns)-math.ceil(len(data.columns)/4), len(data.columns)-math.ceil(len(data.columns)/2))
                print(str(hidden_layer_sizes_MLP))
                #print(feature_list)
                print(len(feature_list))
                
            activation_MLP = ['tanh']#'relu','tanh']#'identity']#,]#]#''],'logistic',]
            solver_MLP = ['lbfgs']#'sgd',,'adam']
            alpha_MLP = np.arange(0.01, 0.05, 1E-3)
            learning_rate_MLP = ['adaptive']#'invscaling']#'constant',, ]
            learning_rate_init_MLP = np.arange(0.1, 1, 1E-5)
            power_t_MLP = np.arange(1, 12, 0.1)
            max_iter_MLP = np.array([500])
            tol_MLP = np.array([1E-3])
            random_state_MLP=[5]
            shuffle_MLP=[False]
            warm_start_MLP = [True]
        
        #SVR
        if(regressor_select == 3):
            C_SVR = np.arange(1,2,1)
            epsilon_SVR = np.arange(1E-6,5E-6,1E-7)
            kernel_SVR = ['rbf', 'sigmoid', 'linear', 'rbf','poly']
            degree_SVR = np.arange(3,10,1)
            gamma_SVR = np.arange(1E-2,1E-3,1E-4)
            coef0_SVR = np.arange(0,0.1,0.1)
            shrinking_SVR = np.array([True])
            tol_SVR = np.arange(0.001,0.01,0.001)
            cache_size_SVR = np.arange(1,10,1)
            verbose_SVR = True
            max_iter_SVR = np.array([-1])
            
         #Bagging regressor
        if(regressor_select == 4):
            base_estimator_Bagging = [RandomForestRegressor(n_estimators = 50,
                                                             random_state = 70)]
            n_estimator_Bagging = np.arange(40, 60, 1)
            random_state_Bagging = np.arange(1,2,1)
            bootstrap_Bagging = np.array([True])
            bootstrap_features_Bagging = np.array([True])
            warm_start_Bagging = np.array([False])
            oob_score_Bagging = np.array([True])
        if(regressor_select == 5):
            n_estimators_RF = np.arange(20,35,1)
            min_samples_leaf_RF = np.arange(4,10,1)
            min_impurity_decrease_RF = np.arange(0.0, 0.2, 0.05)
            max_leaf_nodes_RF = np.arange(18,19,1)
            max_depth_RF = np.arange(100,101,1)
            random_state_RF = np.array([70])

            
        if(regressor_select == 4):
            param_distributions_Bagging = dict(base_estimator  =base_estimator_Bagging, n_estimators = n_estimator_Bagging, 
                                               random_state = random_state_Bagging, bootstrap = bootstrap_Bagging,
                                               bootstrap_features = bootstrap_features_Bagging,
                                               warm_start = warm_start_Bagging,
                                               oob_score = oob_score_Bagging)
        if(regressor_select == 5):
            param_distributions_RF = dict(n_estimators = n_estimators_RF, min_samples_leaf = min_samples_leaf_RF,
                                          min_impurity_decrease = min_impurity_decrease_RF, 
                                          max_leaf_nodes = max_leaf_nodes_RF,
                                          max_depth = max_depth_RF,
                                          random_state = random_state_RF)
        if(regressor_select == 2):
            param_distributions_MLP = dict(hidden_layer_sizes = hidden_layer_sizes_MLP, activation = activation_MLP, 
                                           solver = solver_MLP, alpha = alpha_MLP, learning_rate = learning_rate_MLP, 
                                           learning_rate_init = learning_rate_init_MLP, power_t = power_t_MLP,
                                           max_iter = max_iter_MLP, tol=tol_MLP, random_state=random_state_MLP,
                                           shuffle = shuffle_MLP, warm_start = warm_start_MLP)
        
        if(regressor_select == 1):
            param_distributions_AdaBoost = dict(base_estimator = base_estimator_AdaBoost, loss = loss_AdaBoost, 
                                                learning_rate = learning_rate_AdaBoost, n_estimators = n_estimator_AdaBoost, 
                                                random_state = random_state_AdaBoost)
        
        if(regressor_select == 3):
            param_distributions_SVR = dict(C = C_SVR, epsilon = epsilon_SVR, kernel = kernel_SVR, degree = degree_SVR, gamma = gamma_SVR, 
                                           coef0 = coef0_SVR, shrinking = shrinking_SVR, tol = tol_SVR, cache_size = cache_size_SVR, 
                                           verbose = verbose_SVR, max_iter = max_iter_SVR)
        
        if(regressor_select == 0):
            param_distributions_GradBoost = dict(loss = loss_GradBoost, learning_rate = learning_rate_GradBoost, 
                                                 n_estimators = n_estimators_GradBoost, max_depth = max_depth_GradBoost, 
                                                 criterion = criterion_GradBoost, min_samples_split = min_samples_split_GradBoost,
                                                 subsample = subsample_GradBoost, max_features= max_features_GradBoost,
                                                 max_leaf_nodes = max_leaf_nodes_GradBoost, min_impurity_split = min_impurity_split_GradBoost,
                                                 alpha = alpha_GradBoost, verbose = verbose_GradBoost, warm_start = warm_start_GradBoost,
                                                 random_state = random_state_GradBoost, presort = presort_GradBoost)
       
        if(regressor_select == 4):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_Bagging, n_iter=10)
        
        if(regressor_select == 2):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_MLP, n_iter=30)
        
        if(regressor_select == 1):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_AdaBoost, n_iter=10)
        
        if(regressor_select == 0):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_GradBoost, n_iter=30)
        
        if(regressor_select == 3):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_SVR, n_iter=10)
        
        if(regressor_select == 5):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_RF, n_iter=10)
        if(regressor_select == 6):
            grid = RandomizedSearchCV(estimator=lm, param_distributions = param_distributions_RF, n_iter=10)
        
        grid.fit(X_train, y_train)
    
        grid.best_params_
    
        #lm = SVR(kernel='rbf', C = grid.best_params_['C'], degree=grid.best_params_['degree'], epsilon=grid.best_params_['epsilon'],
         #        gamma=grid.best_params_['gamma'])
        #lm = linear_model.Lasso(alpha = grid.best_params_['alpha'])
        #lm = linear_model.LinearRegression()
        #lm = tree.DecisionTreeRegressor()
        if(regressor_select == 4):
            lm = ensemble.BaggingRegressor( base_estimator = grid.best_params_['base_estimator'], 
                                  n_estimators = grid.best_params_['n_estimators'], 
                                  random_state = grid.best_params_['random_state'],
                                  bootstrap = grid.best_params_['bootstrap'],
                                  bootstrap_features = grid.best_params_['bootstrap_features'],
                                  warm_start = grid.best_params_['warm_start'],
                                  oob_score = grid.best_params_['oob_score'])
            
        if(regressor_select == 3):
            lm = SVR(C = grid.best_params_['C'], epsilon = grid.best_params_['epsilon'], 
                     kernel = grid.best_params_['kernel'], degree = grid.best_params_['degree'],
                     gamma = grid.best_params_['gamma'], coef0 = grid.best_params_['coef0'],
                     shrinking = grid.best_params_['shrinking'], tol = grid.best_params_['tol'],
                     cache_size = grid.best_params_['cache_size'], verbose = grid.best_params_['verbose'],
                     max_iter = grid.best_params_['max_iter'])
        
        if(regressor_select == 0):
            lm = ensemble.GradientBoostingRegressor(loss = grid.best_params_['loss'], learning_rate = grid.best_params_['learning_rate'], 
                                                    n_estimators = grid.best_params_['n_estimators'], max_depth = grid.best_params_['max_depth'],
                                                    criterion = grid.best_params_['criterion'], min_samples_split = grid.best_params_['min_samples_split'],
                                                    subsample = grid.best_params_['subsample'], max_features = grid.best_params_['max_features'],
                                                    max_leaf_nodes = grid.best_params_['max_leaf_nodes'], min_impurity_split = grid.best_params_['min_impurity_split'],
                                                    alpha = grid.best_params_['alpha'], verbose = grid.best_params_['verbose'], warm_start = grid.best_params_['warm_start'],
                                                    random_state = grid.best_params_['random_state'], presort = grid.best_params_['presort'])
        
        if(regressor_select == 2):
            lm = neural_network.MLPRegressor(hidden_layer_sizes = grid.best_params_['hidden_layer_sizes'], activation= grid.best_params_['activation'], 
                                                    solver = grid.best_params_['solver'], alpha = grid.best_params_['alpha'],
                                                    learning_rate = grid.best_params_['learning_rate'], learning_rate_init = grid.best_params_['learning_rate_init'],
                                                    power_t = grid.best_params_['power_t'],
                                                    max_iter = grid.best_params_['max_iter'],
                                                    tol=grid.best_params_['tol'],
                                                    random_state=grid.best_params_['random_state'],
                                                    shuffle = grid.best_params_['shuffle'])
        if(regressor_select == 5):
            lm = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], 
                                       min_samples_leaf = grid.best_params_['min_samples_leaf'],
                                       min_impurity_decrease = grid.best_params_['min_impurity_decrease'], 
                                       max_leaf_nodes = grid.best_params_['max_leaf_nodes'],
                                       max_depth = grid.best_params_['max_depth'],
                                       random_state = grid.best_params_['random_state'])
        
        """
        #lm = ensemble.BaggingRegressor(n_estimators = grid.best_params_['n_estimators'], max_samples = grid.best_params_['max_samples'],
            #max_features = grid.best_params_['max_features'])
        #lm = ensemble.RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_depth = grid.best_params_['max_depth'],
        #max_features = grid.best_params_['max_features'])
        """
        if(regressor_select == 1):
            lm = ensemble.AdaBoostRegressor(base_estimator = grid.best_params_['base_estimator'], 
                                           n_estimators = grid.best_params_['n_estimators'], 
                                           random_state = grid.best_params_['random_state'],
                                           learning_rate = grid.best_params_['learning_rate'],
                                           loss = grid.best_params_['loss'])
    
        A = X[list(l[v])]
        ar = X[list(l[v])].values
        print(A.columns)
    
        normalisedData = preprocessing.StandardScaler().fit_transform(ar)
    
        pca1 = PCA()
        x_pca_p = pca1.fit_transform(normalisedData)
    
        #Split set into training and test sets
        #X_train, X_test, y_train, y_test = train_test_split(
           #normalised[:,2:], data['Price'][data.shape[0]-len(normalised):], test_size=0.4, random_state=0)
        
        #Split set into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
           normalised_x, normalised_y, test_size=no_days, shuffle=False)
        
        if w == 0:
            #The follwoing filters out Thurdays and Fridays from y_test and y_test_n
            y_test_columns = y_test.columns
            X_test_columns = X_test.columns
            X_train_columns = X_train.columns
            y_train_columns = y_train.columns
            
            for i in range(0,y_test.shape[0]):
                X_test.set_value(X_test.index[i], 'Weekday', X_test.index[i].weekday())
                y_test.set_value(y_test.index[i], 'Weekday', y_test.index[i].weekday())
                
            for i in range(0,y_test.shape[0]):
                X_train.set_value(X_train.index[i], 'Weekday', X_train.index[i].weekday())
                y_train.set_value(y_train.index[i], 'Weekday', y_train.index[i].weekday())
            
            X_test = X_test[(X_test['Weekday'] != 2) & (X_test['Weekday'] != 3) & (X_test['Weekday'] != 4)]  
            
            y_test = y_test[(y_test['Weekday'] != 2) & (y_test['Weekday'] != 3) & (y_test['Weekday'] != 4)]
            
            X_train = X_train[(X_train['Weekday'] != 2) & (X_train['Weekday'] != 3) & (X_train['Weekday'] != 4)]  
            
            y_train = y_train[(y_train['Weekday'] != 2) & (y_train['Weekday'] != 3) & (y_train['Weekday'] != 4)]
            
            X_test = X_test[X_test_columns]
            y_test = y_test[y_test_columns]
            
            X_train = X_train[X_train_columns]
            y_train = y_train[y_train_columns]
            
        #convert dataframe back to series
        y_test = y_test.iloc[:,0]
        y_train = y_train.iloc[:,0]
        
        #starttime
        startime = time.time()
    
        #fit the model
        model = lm.fit(X_train,y_train.values.ravel())#, sample_weight = weights[:len(X_train)])
        lm = model
        
        #feature importance
        """
        print(lm.feature_importances_)
        features_df = pd.DataFrame(data=lm.feature_importances_, index=X_train.columns)
        writer = ExcelWriter('feature_importances.xlsx')
        features_df.to_excel(writer, 'sheet1')
        writer.save()
        """
        
        #plot the mlp loss function
        """
        fig = plt.figure(figsize=(18,15), num=2)
        plt.plot(lm.loss_curve_)
        ax=plt.gca()
        ax.grid()
        ax.set_xlabel('Date')
        ax.set_ylabel('Loss function')
        plt.savefig(file_path_test+'/'+'results_'+hours_str+'_loss_function_'+days_str+'.png', bbox_inches='tight')
        """
        stoptime = time.time()
        
        #training time
        times[0] = stoptime - startime
        
        #create saved models folder if it does not exist
        #foldername = (dt.datetime.now() + timedelta(days=0)).strftime('%d%m%Y')
        if not os.path.exists(folder_saveModels):
            os.mkdir(folder_saveModels)
        
        # save the model to disk
        filename = 'finalized_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'.sav'
        with open(folder_saveModels+'\\'+filename,'wb') as f:
            pickle.dump(model, f)
            
        #save the feature_list to file
        with open(folder_saveModels+'\\'+'feature_list'+days_str+'_Hour_'+hours_str+'_'+model_str+'.txt', 'w') as f:
            for item in feature_list:
                f.write("%s\n" % item)
            
        # load the model from disk
        loaded_model = pickle.load(open(folder_saveModels+'\\'+filename, 'rb'))
        print(loaded_model.score(X_train, y_train))
        print(loaded_model.score(X_test, y_test))
        
        #starttime
        starttime = time.time()

        predictions = lm.predict(X_test)
        predictions_train = lm.predict(X_train)
        #predictions_arr[:,0] = predictions
        
        stoptime = time.time()
        
        times[1] = stoptime - startime
        
        #price_V_unbiased = fci.random_forest_error(lm.estimators_[0], X_train, X_test)
        
        #normalised_y[:75] = predictions_arr[:,0]
        inversed = min_max_scaler_y.inverse_transform(predictions.reshape(-1,1))
        print(inversed.shape)
        
        for i in range(0,len(inversed)):
            if inversed[i][0] > UL:
               inversed[i][0] =  UL*10**((inversed[i][0] - UL )/ UL)
        #predictions_arr[:,0] = inversed[:,0]
        
        #Cross validation
        #score = cross_val_score(model, normalised[:,2:], data['Price'][data.shape[0]-len(normalised):], cv=10, scoring="neg_mean_squared_error")
        score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        
        #print(icount)
        root_MSE_arr[icount] = math.sqrt(-score.mean())
        print("Root Mean Squared Error: %0.2f" % (math.sqrt(-score.mean())))
        
        #if math.sqrt(-score.mean()) < RMSE:
        RMSE = math.sqrt(-score.mean())
        R2_best = lm.score(X_train, y_train) 
        predictions_arr = inversed
        best_params = grid.best_params_
        predictions_train_best = predictions_train
        predictions_train_best = min_max_scaler_y.inverse_transform(predictions_train_best.reshape(-1,1))
           
        num_iter += 1
        R2 = lm.score(X_train, y_train)
        print(lm.score(X_train, y_train))
        
        if (num_iter > 0):
            y_train_array = min_max_scaler_y.inverse_transform(y_train.values.reshape(-1,1))
            y_train_array_1 = np.zeros(len(y_train_array))
            y_test_array = min_max_scaler_y.inverse_transform(y_test.values.reshape(-1,1))
            y_test_array_1 = np.zeros(len(y_test_array))
            
            for i in range(0,len(y_test_array_1)):
                y_test_array_1[i] = y_test_array[i][0]
                
            for i in range(0,len(y_train_array_1)):
                y_train_array_1[i] = y_train_array[i][0]
            
            y_test = pd.Series(data=y_test_array_1, index=X_test.index, name='Price')
            y_train = pd.Series(data=y_train_array_1, index=X_train.index, name='Price')
            
            for i in range(0,y_test.shape[0]):
                if y_test.iloc[i] > UL:
                    y_test.iloc[i] =  UL*10**((y_test.iloc[i] - UL )/ UL)
            
            elapsed_times[0,0] = times[0]
    
            elapsed_times[0,1] = times[1]
            break
    return model, R2_best, RMSE, best_params, predictions_train_best, price_V_unbiased, X_train, y_train, y_test, X_test, predictions_arr, predictions_train_best, elapsed_times

def load_saved_models(X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, w, elapsed_times, UL, foldername, predictions_hours, folder_saveModels):
    #function that loads stored models and predicts prices
    icount = 0
    num_iter = 0
    times = np.zeros(2)
    
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
    
    filename = 'finalized_model_'+days_str+'_Hour_'+hours_str+'_'+model_str+'.sav'
    
    # load the model from disk
    lm = pickle.load(open(folder_saveModels+'\\'+filename, 'rb'))
    print(lm.score(X_test, y_test))
    
    writer = ExcelWriter('X_test26.xlsx')
    X_test.to_excel(writer, 'Sheet1')
    writer.save()
    
    #starttime
    startime = time.time()

    predictions = lm.predict(X_test)
    predictions_hours.append(lm.predict(X_test))
    #predictions_arr[:,0] = predictions
    
    stoptime = time.time()
    
    times[1] = stoptime - startime
    
    #price_V_unbiased = fci.random_forest_error(lm.estimators_[0], X_train, X_test)
    
    #normalised_y[:75] = predictions_arr[:,0]
    inversed = min_max_scaler_y.inverse_transform(predictions.reshape(-1,1))
    print(inversed.shape)
    
    for i in range(0,len(inversed)):
        if inversed[i][0] > UL:
           inversed[i][0] =  UL*10**((inversed[i][0] - UL )/ UL)
    #predictions_arr[:,0] = inversed[:,0]
    
    #Cross validation
    #score = cross_val_score(model, normalised[:,2:], data['Price'][data.shape[0]-len(normalised):], cv=10, scoring="neg_mean_squared_error")
    score = cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")
    
    #print(icount)
    root_MSE_arr[icount] = math.sqrt(-score.mean())
    print("Root Mean Squared Error: %0.2f" % (math.sqrt(-score.mean())))
    
    #if math.sqrt(-score.mean()) < RMSE:
    RMSE = math.sqrt(-score.mean())
    
    predictions_arr = inversed
    
       
    num_iter += 1
    
    if (num_iter > 0):
            y_train_array = min_max_scaler_y.inverse_transform(y_train.values.reshape(-1,1))
            y_train_array_1 = np.zeros(len(y_train_array))
            y_test_array = min_max_scaler_y.inverse_transform(y_test.values.reshape(-1,1))
            y_test_array_1 = np.zeros(len(y_test_array))
            
            for i in range(0,len(y_test_array_1)):
                y_test_array_1[i] = y_test_array[i][0]
                
            for i in range(0,len(y_train_array_1)):
                y_train_array_1[i] = y_train_array[i][0]
            
            y_test = pd.Series(data=y_test_array_1, index=X_test.index, name='Price')
            y_train = pd.Series(data=y_train_array_1, index=X_train.index, name='Price')
            
            for i in range(0,y_test.shape[0]):
                if y_test.iloc[i] > UL:
                    y_test.iloc[i] =  UL*10**((y_test.iloc[i] - UL )/ UL)
            
            elapsed_times[0,0] = times[0]
    
            elapsed_times[0,1] = times[1]
            
    return RMSE, predictions_train_best, price_V_unbiased, X_train, y_train, y_test, X_test, predictions_arr, predictions_train_best, elapsed_times, lm

def evaluation_metrics(test_MAPE, test_RMSE, hour, iterations, array1, array2, array3, array4, df, no_days):
    """This function calulated evaluation metircs"""
    
    for d in range(0,len(array1.index)):
        array3[iterations][d] = array2[d][0]
        array4[iterations][d] = array1.iloc[d]

    #Mean Absolute percentage error (MAPE) calculation
    # Minus 2 below to not include errors on forecast prices.
    mape_par = np.zeros(len(df))
    mape_par_normalised = np.zeros(len(df))
    
    for i in range(iterations,iterations+1):
        for j in range(0,len(df)):
            if array4[i][j] != 0:
                mape_par[j] += np.abs((array4[i][j] - array3[i][j]) / array4[i][j])
                mape_par_normalised[j] += np.abs((array4[i][j] - array3[i][j]) / array4[i][j]) / np.mean(array4[i])
            else:
                mape_par[j] += 0 #np.abs((array4[i][j] - array3[i][j]) / 1E-6)
                mape_par_normalised[j] += 0 #np.abs((array4[i][j] - array3[i][j]) / 1E-6) / 1E-6
            
    array_test_MAPE = np.zeros([no_days,2])
    array_test_RMSE = np.zeros([no_days,2])
    
    array_test_MAPE[0,0] = np.mean(mape_par)*100
    array_test_MAPE[0,1] = np.mean(mape_par_normalised)*100
    #array_test_RMSE[0] = math.sqrt(mean_squared_error(df, array3[iterations]))# / np.mean(y_test['Price'])) )*100
    try:
        array_test_RMSE[0,1] = math.sqrt(mean_squared_error(df, array3[iterations])) / np.mean(df)
    except:
        array_test_RMSE[0,1] = 0#math.sqrt(mean_squared_error(df, array3[iterations])) / 1E-6
        
    test_MAPE.append(array_test_MAPE) #[hour,0] = np.mean(mape_par)*100
    #test_MAPE.append() #[hour,1] = np.mean(mape_par_normalised)*100
    test_RMSE.append(array_test_RMSE) #[hour,0] = math.sqrt(mean_squared_error(df, array3[hour]))# / np.mean(y_test['Price'])) )*100
    #test_RMSE.append() #[hour,1] =  math.sqrt(mean_squared_error(df, array3[hour])) / np.mean(df)
    print(np.mean(test_MAPE))
    
    #calculate average test MAPE
    m = 0
    for i in range(0,len(test_MAPE)):
        m += test_MAPE[i][0]
    #print test MAPE to screen    
    print(m/len(test_MAPE))
    
    return test_MAPE, test_RMSE, array3

def plotpercentdiff(R2_best, file_path_train, file_path_test, X_train, X_test, y_train, y_test, regressor_select, test_size, price_V_unbiased, hour, iterations ,predictions_arr, predictions_train_best, train_MAPE, train_RMSE ,test_MAPE, test_RMSE, model_str):
    """ """
    RMSE = 0
    
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
    elif(regressor_select == 6):
        model_str = 'GRNN'
        
    
    ar1 = y_test.values.reshape(len(X_test),1)#**(1/-0.5)
    
    ar1 = ar1.reshape(len(ar1),1)
    
    ar1 = np.concatenate(ar1, axis=0)
     
    ar2 = predictions_arr#**(1/-0.5)
    
    ar2 = np.concatenate(ar2, axis=0)

    ar3 = np.subtract(ar2, ar1)

    ar4 = np.divide(ar3, ar1) 
    
    predictions_train_best = np.concatenate(predictions_train_best, axis=0)
    
    y_bias = np.var(predictions_train_best.reshape(len(predictions_train_best),1), axis=1)
    y_var = (y_train.as_matrix() - np.mean(predictions_train_best.reshape(len(predictions_train_best),1),axis=1))**2
    
    "This function plots the percentage difference between predicted and actual prices"
    #bar chart of percentage differences between actual and predicted prices
    fig = plt.figure(figsize=(18,15), num=1)
    """
    plt.subplot(4,1,1)
    ax=plt.gca()
    plt.title('Actual Price and Predicted Price ' ', R2: ' + str(round(R2_best,2)) + ', RMSE: ' + str(round(RMSE,2)) +', ' + days_str + ', Hour: '+ hour_str)
    ax.set_xlabel('Date')
    ax.set_ylabel('Day ahead Market Price')
    #ax.plot(data_dates.values , df['Price'].values, color = 'green', label='actual')
    #ax.bar(data_dates.iloc[len(A)-int(math.ceil(len(A)*0.4)):].values , y.iloc[len(A)-int(math.ceil(len(A)*0.4)):].subtract(predictions_arr[:,0]),
    plt.bar(data_dates[:len(X_train)].values, data['Price'][:len(X_train)], width=0.1, label='actual price', alpha=0.5)
    plt.bar(data_dates[:len(X_train)].values, predictions_arr[:,0], width=0.1, label='predicted price', alpha=0.5)
    plt.grid()
    ax.legend(loc='best')
    plt.subplot(4,1,2)
    ax=plt.gca()
    plt.title('% differece Predcted and Actual Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Day ahead Market Price')
    ax.bar(data_dates[:len(X_train)].values , ar4[:len(X_train)] * 100, alpha=0.5, color = 'blue', width=0.02, label='% diff predicted - actual')
    ax.set_ylim([-100,100])
    plt.grid()
    ax.legend(loc='best')
    plt.tight_layout()
    formattedy = ["%.0f" % member for member in ar4*100]
    for i, v in enumerate(ar4):
        ax.text(data_dates[i], (v*100)+(abs(v)/v)*0.2, formattedy[i], color='black', ha='center', fontweight='light', fontsize='10')
    """
    pcent_diff = ar4
    
    plt.subplot(2,1,1)
    ax=plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Day ahead Market Price')
    plt.title('Actual Price and Predicted Price ' ', R2: ' + str(round(R2_best,2)) + ' MAPE: '+str(test_MAPE[iterations][0])+' , RMSE: ' + str(round(RMSE,2)) +', ' + days_str + ', Hour: '+ hours_str + ', Explained variance: '+str(round(explained_variance_score(y_test, predictions_arr),2)))
    ax.bar(ind, y_test, label='actual price', alpha=0.5)
    ax.bar(ind, predictions_arr[:,0], label='predicted price', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    #plt.tight_layout()
    plt.grid()
    
    plt.subplot(2,1,2)
    ax=plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('% difference')
    ax.bar(ind, pcent_diff * 100, label = '% difference')
    formattedy = ["%.0f" % member for member in pcent_diff*100]
    for i, v in enumerate(pcent_diff):
        if v == 0:
            ax.text(ind[i], (v*100)+0.2, formattedy[i], color='black', ha='center', fontweight='light', fontsize='10')
        else:
            ax.text(ind[i], (v*100)+(abs(v)/v)*0.2, formattedy[i], color='black', ha='center', fontweight='light', fontsize='10')
    ax.set_ylim([-50,50])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    
    #plt.savefig(file_path_test+'/'+'results_'+hours_str+'_predict_lagPrices_'+days_str+'.png', bbox_inches='tight')
    
    plt.close()
    fig = plt.figure(figsize=(18,15), num=2)
    plt.subplot(2,1,1)
    ax=plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Day ahead Market Price')
    plt.title('Actual Price and Predicted Price ' ', R2: ' + str(round(R2_best,2)) + ', RMSE: ' + str(round(RMSE,2)) +'MAPE:' +str(train_MAPE[iterations][0])+ ', ' + days_str + ', Hour: '+ hours_str)
    ax.bar(ind1, y_train, label='actual price', alpha=0.5)
    ax.bar(ind1, predictions_train_best, label='predicted price', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    
    #calc percentage difference for training set
    ar1_train = y_train.values.reshape(len(X_train),1)#**(1/-0.5)
    ar1_train = ar1_train.reshape(len(ar1_train),1)
    ar2_train = predictions_train_best.reshape(len(predictions_train_best),1)#**(1/-0.5)

    ar3_train = np.subtract(ar2_train, ar1_train)

    ar4_train = np.divide(ar3_train, ar1_train) 
    
    ar4_train = np.concatenate(ar4_train, axis=0)
    
    plt.subplot(2,1,2)
    ax=plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('% difference')
    ax.bar(ind1, ar4_train * 100, label = '% difference')
    formattedy = ["%.0f" % member for member in ar4_train*100]
    for i, v in enumerate(ar4_train):
        if v == 0:
            ax.text(ind1[i], (v*100)+0.2, formattedy[i], color='black', ha='center', fontweight='light', fontsize='10')
        else:
            ax.text(ind1[i], (v*100)++(abs(v)/v)*0.2, formattedy[i], color='black', ha='center', fontweight='light', fontsize='10')
    ax.set_ylim([-25,25])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    
    #plt.savefig(file_path_train+'/'+'results_'+hours_str+'_predict_lagPrices_'+days_str+'.png', bbox_inches='tight')
    
    plt.close()
    #plots pump and interconnector features
    """
    fig = plt.figure(figsize=(18,15), num=3)
    for i in range(0,X_test.shape[1]):
        plt.subplot(X_test.shape[1]+1,1,i+1)
        ax=plt.gca()
        ax.plot(ind, normalised_x.iloc[:len(X_test),i], 'o-', label=data.columns[i+1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        fig.autofmt_xdate()
        plt.grid()
        ax.legend(loc='best')
    """
    #plt.plot(data_dates[:].values, normalised_x[:,1], label=data.columns[2])
    #plt.plot(data_dates[:].values, normalised_x[:,2], label=data.columns[3])
    #plt.plot(data_dates[:].values, normalised_x[:,3], label=data.columns[4])
    #plt.plot(data_dates[:].values, normalised_x[:,4], label=data.columns[5])
    #plt.plot(data_dates[:].values, normalised_x[:,5], label=data.columns[6])
    #plt.plot(data_dates[:].values, normalised_x[:,6], label=data.columns[7])
    """
    plt.subplot(X_test.shape[1]+1,1,X_test.shape[1]+1)
    ax=plt.gca()
    ax.plot(ind, y_test, 'o-', label='Price')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()
    plt.grid()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.suptitle('Price Features : ' + hours_str + ', ' + days_str)
    plt.subplots_adjust(left=0, wspace=0, top=0.97)
    """
    #plt.savefig(file_path_test+'/'+'features_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_predict_lagPrices_'+days_str+'.png', bbox_inches='tight')
    
    plt.close()
    """
    fig = plt.figure(figsize=(18,15), num=4)
    for i in range(0,X_train.shape[1]):
        plt.subplot(X_train.shape[1]+1,1,i+1)
        ax=plt.gca()
        ax.plot(ind1, X_train.iloc[:len(X_train),i], 'o-', label=data.columns[i+1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
        fig.autofmt_xdate()
        plt.grid()
        ax.legend(loc='best')
    """
    #plt.plot(data_dates[:].values, normalised_x[:,1], label=data.columns[2])
    #plt.plot(data_dates[:].values, normalised_x[:,2], label=data.columns[3])
    #plt.plot(data_dates[:].values, normalised_x[:,3], label=data.columns[4])
    #plt.plot(data_dates[:].values, normalised_x[:,4], label=data.columns[5])
    #plt.plot(data_dates[:].values, normalised_x[:,5], label=data.columns[6])
    #plt.plot(data_dates[:].values, normalised_x[:,6], label=data.columns[7])
    """
    plt.subplot(X_train.shape[1]+1,1,X_train.shape[1]+1)
    ax=plt.gca()
    ax.plot(ind1, y_train.iloc[:], 'o-', label='Price')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
    fig.autofmt_xdate()
    plt.grid()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.suptitle('Price Features : ' + hours_str + ', ' + days_str)
    plt.subplots_adjust(left=0, wspace=0, top=0.97)
    """
    #plt.savefig(file_path_train+'/'+'features_'+days_str+'_Hour_'+hours_str+'_'+model_str+'_predict_lagPrices_'+days_str+'.png', bbox_inches='tight')
    
    plt.close()
    fig = plt.figure(figsize=(18,15), num=5)
    plt.subplot(1,1,1)
    ax=plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Day ahead Market Price')
    plt.title('Actual Price and Predicted Price ' ', R2: ' + str(round(R2_best,2)) +'MAPE:' +str(train_MAPE[iterations][0])+ ', RMSE: ' + str(round(RMSE,2)) +', ' + days_str + ', Hour: '+ hours_str)
    ax.bar(ind1, y_bias, label='actual price', alpha=0.5)
    ax.bar(ind1, y_var, label='predicted price', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    
    x = np.arange(0,80)
    y = np.arange(0,80)
    
    plt.close()
    fig = plt.figure(figsize=(18,15), num=6)
    plt.subplot(1,1,1)
    ax=plt.gca()
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    plt.title('Actual Price and Predicted Price ' ', R2: ' + str(round(R2_best,2)) +'MAPE:' +str(test_MAPE[iterations][0])+ ' , RMSE: ' + str(round(RMSE,2)) +', ' + days_str + ', Hour: '+ hours_str)
    ax.scatter(y_test_plot[0], predictions_plot_test[0], label=str(hour))
    #ax.plot(x, y, '--')
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date_train))
    #fig.autofmt_xdate()
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    
    mean_squared_error(y_test_plot[iterations], predictions_plot_test[iterations])
    
    #Mean Absolute percentage error (MAPE)
    #MAPE = np.mean(np.abs((y_test_plot - predictions_plot) / y_test_plot)) * 100
    
    #plt.savefig(file_path_train+'/'+'Pred_Actual_'+days_str+'_AllHours_MAPE'+str(MAPE[hour][0])+model_str+test_size+'_unnormalised_predict_lagPrices_1.png', bbox_inches='tight')
    
    plt.close()
    return model_str

def results_tofile(predictions_plot_test, y_test_plot, no_days, elapsed_times, w, train_MAPE, train_RMSE, test_MAPE, test_RMSE, model_str, start_time, foldername, feature_list, arr4, Hours_arr, models_arr, RMSE_arr, file, l_actual, l_forecast, mode):
    """Writes predicted, actual and MAPE for each hour to excel"""
    y_date = np.zeros((y_test.shape[0],2), dtype='object')
    
    for i in range(0,y_test.shape[0]):
       y_date[i,0] = y_test.index[i].strftime('%d-%m-%Y %H:%M:%S') 
       y_date[i,1] = y_test.index[i].time().strftime('%H:%M:%S')
       
    errors = np.zeros((len(y_test_plot[0]),1))
    
    #Error in each time slot
    if mode == 0:   
        for i in range(0,len(errors)):
            errors[i] = (math.fabs(predictions_plot_test[0][i]-y_test_plot[0][i]) / y_test_plot[0][i])*100
    
    results = np.transpose(np.concatenate((predictions_plot_test, y_test_plot), axis=0))
    
    results = np.concatenate((results, y_date), axis=1)
    
    results = np.concatenate((results, errors), axis=1)
    
    results = np.concatenate((results, train_MAPE[0]), axis=1)
    
    results = np.concatenate((results, train_RMSE[0]), axis=1)
    
    results = np.concatenate((results, test_MAPE[0]), axis=1)
    
    results = np.concatenate((results, test_RMSE[0]), axis=1)

    results = np.concatenate((results, elapsed_times), axis=1)
    
    #array that will contain average values of evaluation metrics
    arr1 = np.zeros(results.shape[1] ,dtype=object)
    arr2 = np.zeros(results.shape[1], dtype=object)
    
    #array that will contain features list
    arr3 =  np.asarray(feature_list)
    
    for i in range(len(arr1)-11,len(arr1)):
        arr1[i] = np.mean(results[:,i])
        arr2[i] = np.std(results[:,i])
    
    arr1[len(arr1)-12] = 'Average'
    arr2[len(arr1)-12] = 'STDEV' 
    
    results = np.concatenate((results, arr1.reshape(1,results.shape[1])), axis=0)
    results = np.concatenate((results, arr2.reshape(1,results.shape[1])), axis=0)
    results = np.concatenate((results, arr4.reshape(results.shape[0],1)), axis=1)
    results = np.concatenate((results, Hours_arr.reshape(results.shape[0],1)), axis=1)
    results = np.concatenate((results, models_arr.reshape(results.shape[0],1)), axis=1)
    results = np.concatenate((results, RMSE_arr.reshape(results.shape[0],1)), axis=1)
    
    results_df = pd.DataFrame(data=results)
    
    l = []
    
    for i in range(0,2*iteration+2):
        for j in range(0, 1):
            print((y_test.index[j].date()).strftime('%Y-%m-%d'))
            if i == 0:
                l.append((y_test.index[j].date()).strftime('%Y-%m-%d')+' Forecast')
            else:
                l.append((y_test.index[j].date()).strftime('%Y-%m-%d')+' Actual')
    
    l.append('Date')
    l.append('Time')
    l.append('% Error')
    l.append('train_MAPE')
    l.append('train_nMAPE')
    l.append('train_RMSE')
    l.append('train_nRMSE')
    l.append('test_MAPE')
    l.append('test_nMAPE')
    l.append('test_RMSE')
    l.append('test_nRMSE')
    l.append('Elapsed time Train')
    l.append('Elapsed time Test')
    l.append('features')
    l.append('Hour')
    l.append('Model')
    l.append('RMSE')
    
    results_df.columns = l
    
    if w == 0:
        filename = 'results_weekend_'+str(start_date.date())+'_'+str(end_date.date())+'_'+model_str+'_'+str(round(arr1[4],2))+'_'+start_time+'_'+str(no_days)+'_'+file+'.xlsx'
    elif w == 1:
        filename = 'results_weekdays_'+str(start_date.date())+'_'+str(end_date.date())+'_'+model_str+'_'+str(round(arr1[4],2))+'_'+start_time+'_'+str(no_days)+'_'+file+'.xlsx'
    elif w == 2:
        filename = 'results_alldays_'+str(start_date.date())+'_'+str(end_date.date())+'_'+model_str+'_'+str(round(arr1[4],2))+'_'+start_time+'_'+str(no_days)+'_'+file+'.xlsx'
    
    """
    writer = ExcelWriter(file_path_train+'/'+filename)
    results_df.to_excel(writer, 'Sheet1')
    writer.save()
    """
    
    #save results to folder forecast results
    #writer = ExcelWriter('Forecast_Results/'+foldername+'/'+filename, engine='openpyxl')
    #results_df.to_excel(writer, 'All Results')
    #writer.save()
    
    
    df_forecast = pd.DataFrame(data=results_df.loc[:results_df.shape[0]-3, results_df.columns[0]])
    
    df_forecast.index = results_df['Date'].loc[:no_days-1]
    
    #df_forecast.to_excel(writer, sheet_name = 'Forecast Prices')
    #writer.save()
    l_forecast.append(df_forecast)
    
    df_actual = pd.DataFrame(data=results_df.loc[:results_df.shape[0]-3, results_df.columns[1]])
    
    df_actual.index = results_df['Date'].loc[:no_days-1]
    
    #df_actual.to_excel(writer, sheet_name = 'Actual Prices')
    #writer.save()
    
    l_actual.append(df_actual)
    
    
    return 

def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N-1)
    #print(thisind)
    #print(x)
    #print(thisind)
    #print(r.date[thisind].strftime('%Y-%m-%d %H:%M:%S'))
    return r.Date[thisind]

def format_date_train(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N1-1)
    #print(thisind)
    #print(x)
    #print(thisind)
    #print(r.date[thisind].strftime('%Y-%m-%d %H:%M:%S'))
    return s.Date[thisind]

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up

def calc_testsize(start_date_tt, end_date_tt, days_type, no_days):
    """function that calculates the size of the test set"""
    time_delta = (end_date_tt - start_date_tt)
    count_days = 0
    date = start_date_tt
    for i in range(0,time_delta.days):
        date += dt.timedelta(days = 1)
        if (date.weekday() in [0,1,2,3,4]):
            count_days += 1
            
    t_size = no_days / count_days
    
    return t_size

def scorer(network, X, y):
    result = network.predict(X)
    return estimators.rmsle(result, y)

def smoothness_holdout(X_train, y_train, X_train_n, y_train_n, min_max_scaler_y):
    """A function that selects the best smoothness parameter for GRNN using the Holdout method"""
    X_train_temp = X_train
    
    sigma = np.array([])    
    sigma = np.linspace(0.000001,50,100)   #sigma = np.linspace(0.05,1,50) for all days and weekdays forecasts
    RMSE_holdout = np.zeros([len(sigma),1])
        
    #leave out sample i in X_train
    for i in range(0,len(sigma)):
        for j in range(0,len(X_train)):
        
            X_train_temp = X_train_n[~X_train_n.index.isin([X_train_n.index[j]])]
            y_train_temp = y_train_n[~y_train_n.index.isin([y_train_n.index[j]])]
                
            grnn = algorithms.GRNN(std = sigma[i])
            
            grnn.train(X_train_temp, y_train_temp)
                    
            predictions = grnn.predict(X_train_n[j:j+1])
            
            if math.isnan(predictions[0][0]) == True:
                RMSE_holdout[i] = 1E6
                break
                #predictions[0][0] = 0
                
            y_pred = min_max_scaler_y.inverse_transform(predictions)
            
            RMSE_holdout[i] =+ math.sqrt(sklearn.metrics.mean_squared_error(np.array([y_train.iloc[j]]), y_pred))
        
            #find index of minimum of RMSE i and return sigma[i]  
    return sigma[np.argmin(RMSE_holdout, axis=0)]

def traintestalgorithm_GRNN(data, X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, no_days, starttime, stoptime, elapsed_times, w, UL):
    """This function trains and predicts with A neupy GRNN"""
    #arrays to hold training and test times
    times = np.zeros(2)
    
    """Take rows in X_train and Y_train that have a positive  Turbine generation"""
    
    X_train, X_test, y_train, y_test = train_test_split(
           data.iloc[:,1:], data.iloc[:,0], test_size=no_days, shuffle=False)
    
    for i in range(0,y_train.shape[0]):
        if y_train[i] > UL:
            y_train[i] = UL + UL*math.log10( y_train[i] / UL)
    
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
           normalised_x, normalised_y, test_size=no_days, shuffle=False)
    
    if w == 0:
        #The follwoing filters out Thurdays and Fridays from y_test and y_test_n
        y_test = y_test.to_frame()
        y_train = y_train.to_frame()
        y_test_columns = y_test.columns
        X_test_columns = X_test.columns
        X_test_n_columns = X_test.columns
        y_test_n_columns = y_test_n.columns
        X_train_columns = X_train.columns
        y_train_columns = y_train.columns
        X_train_n_columns = X_train_n.columns
        y_train_n_columns = y_train_n.columns
        
        for i in range(0,y_test.shape[0]):
            X_test.set_value(X_test.index[i], 'Weekday', X_test.index[i].weekday())
            X_test_n.set_value(X_test_n.index[i], 'Weekday', X_test_n.index[i].weekday())
            y_test.set_value(y_test.index[i], 'Weekday', y_test.index[i].weekday())
            y_test_n.set_value(y_test_n.index[i], 'Weekday', y_test_n.index[i].weekday())
            
        for i in range(0,y_test.shape[0]):
            X_train.set_value(X_train.index[i], 'Weekday', X_train.index[i].weekday())
            y_train.set_value(y_train.index[i], 'Weekday', y_train.index[i].weekday())
            X_train_n.set_value(X_train_n.index[i], 'Weekday', X_train_n.index[i].weekday())
            y_train_n.set_value(y_train_n.index[i], 'Weekday', y_train_n.index[i].weekday())
        
        X_test = X_test[(X_test['Weekday'] != 2) & (X_test['Weekday'] != 3) & (X_test['Weekday'] != 4)]  
        X_test_n = X_test_n[(X_test_n['Weekday'] != 2) & (X_test_n['Weekday'] != 3) & (X_test_n['Weekday'] != 4)] 
    
        y_test = y_test[(y_test['Weekday'] != 2) & (y_test['Weekday'] != 3) & (y_test['Weekday'] != 4)]
        y_test_n = y_test_n[(y_test_n['Weekday'] != 2) & (y_test_n['Weekday'] != 3) & (y_test_n['Weekday'] != 4)]
        
        X_train = X_train[(X_train['Weekday'] != 2) & (X_train['Weekday'] != 3) & (X_train['Weekday'] != 4)]  
        X_train_n = X_train_n[(X_train_n['Weekday'] != 2) & (X_train_n['Weekday'] != 3) & (X_train_n['Weekday'] != 4)] 
    
        y_train = y_train[(y_train['Weekday'] != 2) & (y_train['Weekday'] != 3) & (y_train['Weekday'] != 4)]
        y_train_n = y_train_n[(y_train_n['Weekday'] != 2) & (y_train_n['Weekday'] != 3) & (y_train_n['Weekday'] != 4)]
        
        X_test = X_test[X_test_columns]
        X_test_n = X_test_n[X_test_n_columns]
        y_test = y_test[y_test_columns]
        y_test_n = y_test_n[y_test_n_columns]
        
        X_train = X_train[X_train_columns]
        X_train_n = X_train_n[X_train_n_columns]
        y_train = y_train[y_train_columns]
        y_train_n = y_train_n[y_train_n_columns]
        
        #convert dataframe back to series
        y_test = y_test.iloc[:,0]
        y_train = y_train.iloc[:,0]
    
    std_holdout = smoothness_holdout(X_train, y_train, X_train_n, y_train_n, min_max_scaler_y)
    
    print(std_holdout)
    #grnn = algorithms.GRNN()
    #grnn.train(X_train_n, y_train_n)
    #error = scorer(grnn, X_test_n, y_test_n)
    
    #random_search = RandomizedSearchCV(
    #    estimator = grnn,
    #    param_distributions={'std': np.arange(0.01, 10, 0.01)},
    #    n_iter=400,
    #    scoring=scorer
    #)
        
    #random_search.fit(X_train_n, y_train_n)
        
    #grnn = algorithms.GRNN(std = random_search.best_params_['std'])
    grnn = algorithms.GRNN(std = std_holdout)
    
    starttime = time.time()
    
    grnn.train(X_train_n, y_train_n)
    
    stoptime = time.time()
    
    #training time
    times[0] = stoptime - starttime
    
    starttime = time.time()
    
    predictions_arr = grnn.predict(X_test_n)
    
    stoptime = time.time()
    
    times[1] = stoptime - starttime
        
    elapsed_times[0,0] = times[0]
    
    elapsed_times[0,1] = times[1]
    
    predictions_arr = min_max_scaler_y.inverse_transform(predictions_arr)
    
    for i in range(0,y_train.shape[0]):
        if y_train[i] > UL:
            y_train[i] = UL*10**((y_train[i] - UL )/ UL)
            
    for i in range(0,len(predictions_arr)):
        if predictions_arr[i][0] > UL:
           predictions_arr[i][0] =  UL*10**((predictions_arr[i][0] - UL )/ UL)
    
    
    
    predictions_train_best = grnn.predict(X_train_n)
    
    predictions_train_best = min_max_scaler_y.inverse_transform(predictions_train_best)
    
    #calc RMSE
    estimators.rmse(predictions_arr, y_test_n)
    
    return predictions_train_best, predictions_train, predictions_arr, X_train, y_train, X_test, y_test, starttime, stoptime, elapsed_times

def select_features_based_on_covariance(data, feature_list, end_date_tt):
    #Select predictor variables using covariance 
    data = data.fillna(0)
    
    cov_data = np.corrcoef(data.iloc[:,1:].T)
    
    features_cutoff = [i+1 for i,e in enumerate(cov_data[0]) if math.fabs(e)>=0.0]
    
    feature_list_1 = []
    feature_list_1 = [i for j, i in enumerate(feature_list) if j in features_cutoff]
    
    feature_list_2 = ['Date']
    for i in range(0,len(feature_list_1)):
        feature_list_2.append(feature_list_1[i])
    
    """
    #add lags from all other variables the same as highly correlated variables  
    feature_list_3 = ['Date', 'Price']
    for i in range(2,len(feature_list)):
        for j in range(2,len(feature_list_2)):
            if ((feature_list[i].split()[1]) == (feature_list_2[j].split()[1])):
                feature_list_3.append(feature_list[i])
    
    """
    #'Date' and 'Price' is already in feature_list, if there are more then change feature_list
    if len(feature_list_2) > 2:   
        feature_list = feature_list_2
    
    data = data[feature_list]
    
    return data, feature_list

def remove_highWinddays(data):
    l = []
    for i in range(0,data.shape[0]):
        if (data['Price'].iloc[i] > 100):
            if data['Date'].iloc[i].date() not in  l:
                l.append(data['Date'].iloc[i].date())
    
    for i in range(0, data.shape[0]):
        data.set_value(data.index[i], 'Date_only', data['Date'].iloc[i].date())
    
    data = data[~data['Date_only'].isin(l)]
    
    data = data.drop(labels='Date_only', axis=1)
    
    return data

def return_feature_list_hour(hour):
    if hour == 0:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn']
    elif hour == 1:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET']
    elif hour == 2:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature']
    elif hour == 3:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff',
                        'operating_reserve_requirement']
    elif hour == 4:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff',
                        'operating_reserve_requirement', 
                        'asp_price_usage','default_price_usage',
                            'administered_scarcity_price','Hour']
    elif hour == 5:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity']
    elif hour == 6:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff',
                        'operating_reserve_requirement']
    elif hour == 7:
        feature_list_hour  = ['Date', 'Price','Price_lag','Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff']
    elif hour == 8:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price']
    elif hour == 9:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price']
    elif hour == 10:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff', 
                        'operating_reserve_requirement', 
                        'asp_price_usage','default_price_usage',
                        'administered_scarcity_price']
    elif hour == 11:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea']
    elif hour == 12:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea']
    elif hour == 13:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all']
    elif hour == 14:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea']
    elif hour == 15:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff']
    elif hour == 16:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea']
    elif hour == 17:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn']
    elif hour == 18:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff', 
                        'operating_reserve_requirement', 
                        'asp_price_usage','default_price_usage',
                        'administered_scarcity_price']
    elif hour == 19:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET']
    elif hour == 20:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature']
    elif hour == 21:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price']
    elif hour == 22:
        feature_list_hour  = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff', 
                        'operating_reserve_requirement', 
                        'asp_price_usage','default_price_usage']
    elif hour == 23:
        feature_list_hour = ['Date', 'Price', 'Price_lag', 'Interconnector_MOYLE', 'Actual_Wind_Generation_all', 'pmea', 
                        'Gas_Price', 'temperature', 
                        'BETTA_Price', 
                        'Interconnector_NET', 'imbalance_price',
                         'total_pn',
                         'load_wind_diff','Actual_System_Demand_all',
                        'total_unit_availability','short_term_reserve_quantity','availability_demand_diff', 
                        'operating_reserve_requirement', 
                        'asp_price_usage','default_price_usage',
                        'administered_scarcity_price','Hour','qpar',
                       'tso_renewable_forecast']
        
    
    return feature_list_hour

def remove_outliersFromTrainSet(data, no_days):
    """This function removes outliers from the training set only """
    col_temp_ws = []
    col_temp_wg = []
    col_temp_wd = []
        
    matching_ws = [s for s in data.columns if "Price" in s]
    #matching_wg = [s for s in data.columns if "WindGust" in s]
    #matching_wd = [s for s in data.columns if "winddir" in s]
        
    for i in range(0,len(matching_ws)):
        col_temp_ws.append(matching_ws[i])
    """    
    for i in range(0,len(matching_wg)):
        col_temp_wg.append(matching_wg[i])
        
    for i in range(0,len(matching_wd)):
        col_temp_wd.append(matching_wd[i])
    """    
    data_train_tmp = data.iloc[:-no_days]
        
    removed_outliers_ws = data_train_tmp[col_temp_ws[0]].between(data_train_tmp[col_temp_ws[0]].quantile(0.05), 
                                     data_train_tmp[col_temp_ws[0]].quantile(0.95))
    """
    removed_outliers_wg = data_train_tmp[col_temp_wg[0]].between(data_train_tmp[col_temp_wg[0]].quantile(0.05), 
                                     data_train_tmp[col_temp_wg[0]].quantile(0.95))

    removed_outliers_wd = data_train_tmp[col_temp_wd[0]].between(data_train_tmp[col_temp_wd[0]].quantile(0.05), 
                                     data_train_tmp[col_temp_wd[0]].quantile(0.95))
    
    data_train_tmp = data_train_tmp[(data_train_tmp.index.isin(removed_outliers_ws[removed_outliers_ws == True].index.tolist()))
        & (data_train_tmp.index.isin(removed_outliers_wg[removed_outliers_wg == True].index.tolist()))
        & (data_train_tmp.index.isin(removed_outliers_wd[removed_outliers_wd == True].index.tolist()))]
    """   
    data_train_tmp = data_train_tmp[(data_train_tmp.index.isin(removed_outliers_ws[removed_outliers_ws == True].index.tolist()))]   
    
    data = pd.concat([data_train_tmp, data.iloc[-no_days:]])
    
    return data

def create_savedModelsFolder(folder_saveModels):
    """This function creates a folder to save models  """
    
    if not os.path.exists(folder_saveModels):
        os.makedirs(folder_saveModels)
        
    return

def reevaluate_windspeed(data):
    """   """
    z0 = 0.1 #roughness parameter
    col = []
    
    matching = [s for s in data.columns if "_speed" in s or "_gust" in s]
                
    for i in range(0,len(matching)):
        col.append(matching[i])
    
    for i in col:
        for j in range(0,data.shape[0]):
            v = data[i].iloc[j]*(math.log(65/z0) / math.log(10/z0))
            data.set_value(data.index[j], i, v)
        
    return data

def remove_skew(data):
    """ This function removes skew from WindSpeed"""
    
    matching = [s for s in data.columns if "Price" in s]
    
    df_pow = data[matching[0]].apply(np.log10)
    
    data['Price'] = df_pow
    
    for i in range(0,data.shape[0]):
        if data['Price'].iloc[i] == -inf:
            data.set_value(data.index[i], 'Price', 0)
    
    return data

def transform_data(data):
    """ This function removes   """
    
    matching = [s for s in data.columns if "WindSpeed" in s]
    
    #
    posdata = data[matching[0]][data[matching[0]]>0]
    
    if (len(posdata) > 0):
        x = np.empty_like(data[matching[0]])
        bcdata, lam = scipy.stats.boxcox(posdata)
        
        x[data[matching[0]]>0] = bcdata
        x[data[matching[0]] == 0] = -1/lam
        
        data[matching[0]] = x
    else:
        data[matching[0]] = data[matching[0]]
    
    matching = [s for s in data.columns if "Price" in s]
    
    #
    posdata = data[matching[0]][data[matching[0]]>0]
    
    if (len(posdata) > 0):
        y = np.empty_like(data[matching[0]])
        bcdata, lam = scipy.stats.boxcox(posdata)
        
        y[data[matching[0]]>0] = bcdata
        y[data[matching[0]] == 0] = -1/lam
        
        data[matching[0]] = y
    else:
        data[matching[0]] = data[matching[0]]
    
    return data
    
    
if __name__ == '__main__':
    modes = ['training', 'Predict']
    
    mode = modes[int(sys.argv[5])] #modes[1]#


    #number of days to forecast
    no_days = int(sys.argv[2])*24 #48#
    
    #Upper limit prices
    UL = 1E12#int(sys.argv[2])
    #print('Upper limit price' +str(UL))
    
    regressor_select = int(sys.argv[1]) #2#GradBoost = 0, AdaBoost = 1, MLP = 2, SVR = 3, Bagging = 4, RandomForest = 5, GRNN  = 6
    
    elapsed_times = np.zeros((no_days,2))
    #training metrics
    train_MAPE = []
    train_RMSE = []
    #test metrics
    test_MAPE = []
    test_RMSE = []
    starttime = 0
    stoptime = 0
    
    l_forecast = []
    l_actual = []
    
    iterations = 0
    
    if(regressor_select == 0):
        model_str = 'GradBoost'
    elif(regressor_select == 1):
        model_str = 'AdaBoost'
    elif(regressor_select == 2):
        model_str = 'MLP'
    elif(regressor_select == 3):
        model_str = 'SVR'
    elif(regressor_select == 4):
        model_str = 'Bagging'
    elif(regressor_select == 5):
        model_str = 'RandomForest'
    elif(regressor_select == 6):
        model_str = 'GRNN'
    
    
    #start time of the run
    run_start_date = time.strftime("%d-%m-%Y")
    start_time = time.strftime("%d-%m-%Y-%H-%M-%S")
    
    # Price Features
    features_to_add = [
            'net_imbalance_volume', 'imbalance_price', 'asp_price_usage',
            'pmea',
                'qpar',
                'total_pn'
            
              
                        
                   ]
    """
     'BETTA_Price',
                        'Actual_System_Demand_all',
                        'Gas_Price',
                'default_price_usage',
                        'administered_scarcity_price',
                        'short_term_reserve_quantity',
                        'operating_reserve_requirement', 
                        'asp_price_usage',
                'Interconnector_EWIC',
                'total_pn',
                'tso_renewable_forecast',
                'Interconnector_MOYLE',
                'Actual_Wind_Generation_all',
                'temperature',
                'availability_demand_diff',
                'market_backup_price',
                'total_unit_availability',
                'imbalance_price',
                'load_wind_diff',
                'Interconnector_NET',
                'pmea',
                'qpar',
                'Hour', 
              'Weekday1',
              'temperature' 
              
              'Price_lag', 
                               'Actual_System_Demand_all', 
                               'default_price_usage', 'administered_scarcity_price', 'Actual_Wind_Generation_all',
                               'short_term_reserve_quantity', 'operating_reserve_requirement', 
                               'asp_price_usage', 'Interconnector_EWIC', 'total_pn', 'tso_renewable_forecast', 
                               'Interconnector_MOYLE', 'temperature', 
                               'availability_demand_diff', 'market_backup_price', 'total_unit_availability', 
                               'imbalance_price'
    """
    
    
    
    

    for iteration in range(0,1):#len(list(it.combinations(features_to_add,combine)))):
        
        
        for hour in [0]:
            
            #create folder named forecast date that will hold results
            #foldername = (dt.datetime.now() + timedelta(days=0)).strftime('%d%m%Y') #folder named as today's date
            foldername = 'Generator_Forecast_Results'
            if not os.path.exists('C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\Forecast_Results\\'+foldername):
                os.mkdir('C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\Forecast_Results\\'+foldername)
    
            #data = pd.read_excel('ISEM_datafiles/06-11-2018-10-52-23.xlsx')
            #read latest datafile
            
            list_of_files = glob.glob(r'C:\Users\daniel.mcglynn\Documents\WindForecast\Closest_WeatherStn_to_Turbine\DataFile_EachTurbine1\*.xlsx')
            #data = pd.read_excel(max(list_of_files, key=os.path.getmtime))
            #print(max(list_of_files, key=os.path.getmtime))
            #matching_files = [s for s in list_of_files if "2273" in s]
            
            matching_files = [s for s in list_of_files if "1221" in s]
            
            #read all turbine IDs from generators table in windMachineLearning database in to a list
            generator_ids = pd.DataFrame()
            
            connection = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=windMachineLearning;'
                                'Trusted_connection=yes')
            
            generator_ids = pd.read_sql("select [generator_id] from generators", connection)
            
            list_of_generators = generator_ids['generator_id'].tolist()
            
            for file in list_of_generators: # for every CE wind turbine
                
                #read data filterd by generator_id
                data = pd.read_sql("""select * from weather where [generator_id] = ?""", connection, params=[file])
                
                data.columns = ['generator_id', 'Date', 'wind_speed', 'wind_direction', 'wind_gust','Price']
                
                #lists of training and testing results
                y_train_plot = []
                predictions_plot_train = []
                y_test_plot = []
                predictions_plot_test = []
                predictions_hours = []
                
                arr4 = np.zeros(no_days+2, dtype=object)
                Hours_arr = np.zeros(no_days+2)
                models_arr = np.zeros(no_days+2, dtype=object)
                RMSE_arr = np.zeros(no_days+2)
                
                Hours_arr[0] = hour
                
             
                feature_list = []
                
                features = []
                
                #data  = pd.read_excel(file)
                
                data = data.fillna(0)
                
                ####
                
                #open file dialog box
                
                #root = Tk()
                #root.filename =  filedialog.askopenfilename(initialdir = r"C:\Users\daniel.mcglynn\Documents\WindForecast\Closest_WeatherStn_to_Turbine\DataFile_EachTurbine1",title = "Select file",filetypes = (("xlsx", "*.xlsx"),("all files","*.*")))
                
                #data = pd.read_excel(root.filename)
                
                            
                data = data[~data.index.duplicated(keep='first')]
                
                date_price_list = ['Date', 'Price']
                
                #find columns that contain "WindSpeed"
                
                #data = remove_skew(data)
                
                matching = [s for s in data.columns if "wind_" in s]
                
                for i in range(0,len(matching)):
                    date_price_list.append(matching[i])
                #date_price_list = data.columns
                
                for i in range(0,len(date_price_list)):
                    features.append(date_price_list[i])
                    feature_list.append(date_price_list[i])
                
                #data = remove_highWinddays(data)
                
                #data = pd.read_excel('data_2015_2017_SeasonallyAdjustedPrice.xlsx')
                #data = pd.read_excel('SEMO_df_MeanTSD_MeanPrice1.xlsx') #hourly SEMO dataset
                #data = pd.read_excel('SEMO_df_1.xlsx')
                #data = pd.read_excel('SEMO_df_tomorrowsPrice.xlsx')
                
                #data[data['Price'] == 0] = 0.01 
                
                #warm_mask = (data['Date'].map(lambda x: x.month) <= 4) | (data['Date'].map(lambda x: x.month) >= 10)
                #data = data[warm_mask]
                #data = data.reset_index(drop=True)
        
                #folder to save results
                folder_path = r'C:/Users/daniel.mcglynn/Documents/WindForecast/Results_'+str(no_days)+'_Wind_Forecast_'+run_start_date+'/'
                
                folder_saveModels = r''+os.getcwd()+'\Saved_Models'
                
                create_savedModelsFolder(folder_saveModels)
        
                
                ##### select days and hours ######   
                p = 0 # 0 off-peak , 1 peak hours
                w = 2# 0 weekend, 1 weekdays, 2 all days
                
                #Start Date and End Date
                start_date = dt.datetime(2020,1,7,0,0,0)

                try:
                    end_date =  dt.datetime.strptime(str(sys.argv[3])+' '+str(sys.argv[4]), '%Y-%m-%d %H:%M:%S')#dt.datetime(2020,1,30,22,0,0)+timedelta(hours = iteration)#
                except ValueError:
                    print(ValueError)
                #start train set
                start_date_tt = start_date
                end_date_tt = end_date+timedelta(hours = iteration)#(dt.datetime.strptime(str(sys.argv[4]), '%Y-%m-%d')).date() 
                
            
                hour = hour
                
                #t_size = 0.02851 # One day forecast
                #t_size = 0.1428 #One week forecast# One #0.0312 #size of test sample
                hours_str = file
                #Select weekend or weekdays
                if w == 0:
                    days_str = 'weekend'
                elif w == 1:
                    days_str = 'weekdays'
                elif w == 2:
                    days_str = 'all_days'
                
               
                #list of features to be smoothed
                features_to_smooth = []
                print(features_to_smooth)
                
               
                
                lag_price_labels = []
                
                print(data.head())   
                    
                g = np.zeros(150)
                X_train = np.zeros(150)
                y_train = np.zeros(150)
                X_test = np.zeros(150)
                y_test = np.zeros(150)
                
                t_size = no_days#calc_testsize(start_date_tt, end_date_tt, w, no_days)
                
                test_size = str(t_size)
                
                data, hours_str, days_str, data_dates = filter_data(data, hours_str, days_str, feature_list, no_days)
                
                #data = remove_outliersFromTrainSet(data, no_days)
                
                #data = reevaluate_windspeed(data)
                
                #data = transform_data(data)
                
                
                file_path_train, file_path_test = createDirectory(start_date_tt, end_date_tt, regressor_select, no_days, folder_path)
                
                lag_price_labels, feature_list, data = get_lag_prices(data, feature_list, acf, mode)
                
                #data = filter_99pc(data, hour,  acf)
                if mode == 'training':
                    data, feature_list = select_features_based_on_covariance(data, feature_list, end_date_tt)
                
                #data['Gas Price'] = data['Gas Price'].replace(0,np.nan)
                #data['Gas Price'] = data['Gas Price'].ffill()
                """data['Oil'] = data['Oil'].replace(0,np.nan)
                data['Oil'] = data['Oil'].bfill()
                data['Coal'] = data['Coal'].replace(0,np.nan)
                data['Coal'] = data['Coal'].bfill()"""
                
                data = data.ffill()
                data.fillna(0, inplace=True)
                
                data_smoothed = pd.DataFrame(index = data.index, columns = data.columns)
                x1 = pd.DataFrame()
                x2 = pd.DataFrame()
                y = np.zeros((data.shape[0], data.shape[1]))
                normalised_x = np.zeros((data.shape[0], len(feature_list)-2))
                normalised_y = np.zeros((data.shape[0], 1))
                data_dates = pd.DataFrame()
                
                
                
                #filter out the dataset that will be split into training and test sets
                data_columns = data.columns
                
                for i in range(0,data.shape[0]):
                    data.set_value(data.index[i], 'DateOnly', data['Date'][data.index[i]].date()) #create date column
                
                data = data[(((data['Date'] >= start_date_tt) & (data['Date'] <= end_date_tt)))]
                
                data = data.sort_values(by=['Date'])
                
                if mode == 'training':
                    data = data[data_columns]
                elif mode == 'Predict':
                    #read file of feature_list into list
                    fname = r''+os.getcwd()+'\Saved_models/'+'feature_list'+days_str+'_Hour_'+hours_str+'_'+model_str+'.txt'
                    with open(fname) as f:
                        l = f.readlines()
                    l = [x.strip() for x in l]
                    
                    feature_list = l
                    
                    data = data[feature_list]
                
                #data = data.set_index('Date')
                
                #data = data[np.isfinite(data[data.columns[np.argmax(window_sizes)]])]
                #data_dates = data.index
                
                #data, data_dates = smoothvars(data, data_dates, g, feature_list)
                data, data_dates = smoothvars_exponential(data, data_dates, g, feature_list, features_to_smooth)
                #normalised = normalisevars(data)
                normalised_x, normalised_y, min_max_scaler_y, min_max_scaler_x2 = rescaleData(data, x1, x2, y, UL, days_str, hours_str, model_str, mode) #uses MinMaxScaler to rescale dataset
                
                #Exponential smoothing
                #normalised_x = smoothvars_test(normalised_x, feature_list, features_to_smooth)
                
                outlier_rows = findoutliers(normalised_x, normalised_y)
                #outlier_rows = findoutliers_LocalOutlierFactor(normalised_x, normalised_y)
                #clipoutliers(normalised_x, normalised_y)
                weights = set_sample_weights(normalised_x, outlier_rows)
                #weights = set_sample_weights_LOF(normalised_x, outlier_rows)
                X_train, y_train, X_test, y_test, X, y, l, predictions_arr, root_MSE_arr, predictions_train, predictions_train_best, price_V_unbiased = definearrays(normalised_x, normalised_y, X_train, y_train, X_test, y_test, t_size, data, feature_list, w)
                
    
                nn_layer_sizes = [[(1,)]]
        
                
                nn_layer_size = nn_layer_sizes[0]
                
                if mode == 'training':
                    if regressor_select in [0,1,2,3,4,5]:
                        model, R2_best, RMSE, best_params, predictions_train_best, price_V_unbiased, X_train, y_train, y_test, X_test, predictions_arr, predictions_train_best, elased_times = traintestalgorithm(X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, w, elapsed_times, UL, foldername, nn_layer_size, folder_saveModels)
                        models_arr[0] = str(model)
                        RMSE_arr[0]  = RMSE
                    else:
                        predictions_train_best, predictions_train, predictions_arr, X_train, y_train, X_test, y_test, starttime, stoptime, elapsed_times = traintestalgorithm_GRNN(data, X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, no_days, starttime, stoptime, elapsed_times, w, UL)
                elif mode == 'Predict':
                    RMSE, predictions_train_best, price_V_unbiased, X_train, y_train, y_test, X_test, predictions_arr, predictions_train_best, elased_times, lm = load_saved_models(X_train, y_train, X_test, y_test, min_max_scaler_y, normalised_x, normalised_y, t_size, regressor_select, predictions_train_best, weights, price_V_unbiased, w, elapsed_times, UL, foldername ,predictions_hours, folder_saveModels)
                    models_arr[0] = str(lm)
                #append empty arrays that store results 
                y_train_plot.append(np.zeros(len(predictions_train_best)))
                predictions_plot_train.append(np.zeros(len(predictions_train_best))) 
                y_test_plot.append(np.zeros(len(predictions_arr)))
                predictions_plot_test.append(np.zeros(len(predictions_arr))) 
                    
                
                X_test.sort_index(inplace=True)
                y_test.sort_index(inplace=True)
                test_df = pd.DataFrame()
                test_df = y_test.to_frame()
                r = test_df.to_records() #convert dataframe to records array
                
                X_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)
                train_df = pd.DataFrame()
                train_df = y_train.to_frame()
                s = train_df.to_records() #convert dataframe to records array
                
                r.sort()
                s.sort()
                #r = r[-550:]  # get the last 30 days
                
                N = len(X_test)
                ind = np.arange(N)  # the evenly spaced plot indices
                
                N1 = len(X_train)
                ind1 = np.arange(N1)
                
                #calculate error bars
                #err_down, err_up = pred_ints(model, X_train)
                train_MAPE, train_RMSE, predictions_plot_train = evaluation_metrics(train_MAPE, train_RMSE, hour, iterations, y_train, predictions_train_best, predictions_plot_train, y_train_plot, y_train, no_days)
                test_MAPE, test_RMSE, predictions_plot_test = evaluation_metrics(test_MAPE, test_RMSE, hour, iterations, y_test, predictions_arr, predictions_plot_test, y_test_plot, y_test, no_days)
                
                #model_str = plotpercentdiff(0, file_path_train, file_path_test, X_train, X_test, y_train, y_test, regressor_select, test_size, price_V_unbiased, hour, iterations, predictions_arr, predictions_train_best, train_MAPE, train_RMSE, test_MAPE, test_RMSE, model_str)
                
                print("Explained variance score is: " + str(round(explained_variance_score(y_test, predictions_arr),2)))
                
                #iterations += 1
                
                #save the feature_list to file
                with open('C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI\\Saved_Models\\'+'/'+'features_Iteration'+str(iteration)+days_str+'_Hour_'+hours_str+'_'+model_str+'.txt', 'w') as f:
                    for item in features:
                        f.write("%s\n" % item)
                        
                arr4[0] = ', '.join(feature_list)
            
                results_tofile(predictions_plot_test, y_test_plot, no_days, elapsed_times, w, train_MAPE, train_RMSE, test_MAPE, test_RMSE, model_str, start_time, foldername, feature_list, arr4, Hours_arr, models_arr, RMSE_arr, file, l_actual, l_forecast, mode)

    #Save total CE turbine wind generation to a excel file        
    concated_results = saveAllGenResultsToExcel.saveResultsToFile(l_forecast, l_actual, start_date, end_date, start_time, model_str, no_days)
    
    update_resultsTable.update_results(concated_results)
"""
plt.figure(figsize=(18,10), num=1)
ax=plt.gca()
fig = plt.figure()
ax.bar(ind, r.Price)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()
"""