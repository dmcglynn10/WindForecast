# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:49:31 2020

@author: Daniel.McGlynn
"""
import glob
import pandas as pd
import numpy as np
import pypyodbc as db
from pandas import ExcelWriter
import datetime as dt

def update_results(concated_results):
    connection = db.connect('Driver={SQL Server};'
                                    'Server=CLKSQL001\SQL2014;'
                                    'Database=windMachineLearning;'
                                    'Trusted_connection=yes')
    
    cursor = connection.cursor()
    
    df = concated_results#pd.read_excel(r'C:\Users\daniel.mcglynn\Documents\WindForecast\Forecast_Results\Generator_Forecast_Results\final_result\1HR_results\results_alldays_2020-01-07_2020-01-27_MLP_05-02-2020-12-18-50_48_.xlsx', sheet_name='Total Wind Gen')
    
    for i in range(0,df.shape[0]):
        df.set_value(df.index[i], 'datetime', dt.datetime.strptime(df.index[i] , '%d-%m-%Y %H:%M:%S'))
    
    df.columns = ['forecast_results', 'total_metered_generation', 'datetime']
    
    df_weatherTurbinesMerged = df
    counter = 0   
    for index, row in df_weatherTurbinesMerged.iterrows():
        cursor.execute("SELECT count(1) from Results where date = ?", (row['datetime'],))
        no_records = cursor.fetchone()[0]
        if no_records == 1:
            cursor.execute("""UPDATE Results
                              SET [forecast_results] = ? where [date] = ?""",
                              (round(row['forecast_results'],2), row['datetime']))
            connection.commit() 
            print(counter / int(df_weatherTurbinesMerged.shape[0]))
        elif no_records == 0:
            cursor.execute("""INSERT INTO Results([forecast_results] , [total_metered_generation], [date])
                            VALUES (?,?,?)""", (round(row['forecast_results'],2), row['total_metered_generation'], row['datetime']))
            connection.commit() 
            print(counter / int(df_weatherTurbinesMerged.shape[0]))
            counter+=1
            
    cursor.close()
    connection.close()
    
    return

if __name__ =='__main__':
    update_results()