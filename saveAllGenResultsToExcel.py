# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:49:06 2020

@author: Daniel.McGlynn
"""
import pandas as pd
from pandas import ExcelWriter
import glob


def saveResultsToFile(l_forecast, l_actual, start_date, end_date, start_time, model_str, no_days):
    """ This function concats all CE wind turbine forecast and actual wind generation values into a single dataframe
    respectively and saves then into a single Excel file"""
    
    #define lists
    f = []
    a = []
    
    #Forecast Wind Gen concat
    for file in l_forecast:
        
        l = []
    
        df = file
            
        #df.index = df['Date']
    
        matching = [s for s in df.columns if "Forecast" in s]
    
        for i in matching:
            l.append(i)
    
        f.append(df[l])
    
    forecast_windGen = pd.concat(f, axis=1)
    
    #add a column with the sum of each column
    forecast_windGen['forecast_total_Gen'] = forecast_windGen.sum(axis=1)
    
    #Actual Wind Gen concat
    for file in l_actual:
        
        l = []
    
        df = file
        
        #df.index = df['Date']
    
        matching = [s for s in df.columns if "Actual" in s]
    
        for i in matching:
            l.append(i)
    
        a.append(df[l])
    
    actual_windGen = pd.concat(a, axis=1)
    
    #add a column with the sum of each column
    actual_windGen['actual_total_Gen'] = actual_windGen.sum(axis=1)
    
    #assign filename of results file
    filename = 'results_alldays_'+str(start_date.date())+'_'+str(end_date.date())+'_'+model_str+'_'+start_time+'_'+str(no_days)+'_'+'.xlsx'
    
    #save dataframe to an Excel file
    writer = ExcelWriter('C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\Forecast_Results\\Generator_Forecast_Results\\final_result\\'+filename, engine='openpyxl')
    forecast_windGen.to_excel(writer, sheet_name = 'Forecast Prices')
    writer.save()
    
    #Actual prices sheet
    actual_windGen.to_excel(writer, sheet_name = 'Actual Prices')
    writer.save()
    
    #Total Generation forecast page
    concated_results = pd.concat([forecast_windGen['forecast_total_Gen'], actual_windGen['actual_total_Gen']], axis=1)
    concated_results.to_excel(writer, sheet_name = 'Total Wind Gen')
    writer.save()
    
    return concated_results


if __name__ == '__main__':
    
    f = []
    a = []
    
    list_of_files = glob.glob(r'C:\Users\daniel.mcglynn\Documents\WindForecast\Forecast_Results\Generator_Forecast_Results\*.xlsx')
    
    #Forecast Wind Gen concat
    for file in list_of_files:
        
        l = []
    
        df = pd.read_excel(file)
            
        df.index = df['Date']
    
        matching = [s for s in df.columns if "Forecast" in s]
    
        for i in matching:
            l.append(i)
    
        f.append(df[l])
    
    forecast_windGen = pd.concat(f, axis=1)
    
    #add a column with the sum of each column
    forecast_windGen['forecast_total_Gen'] = forecast_windGen.sum(axis=1)
    
    #Actual Wind Gen concat
    for file in list_of_files:
        
        l = []
    
        df = pd.read_excel(file)
        
        df.index = df['Date']
    
        matching = [s for s in df.columns if "Actual" in s]
    
        for i in matching:
            l.append(i)
    
        a.append(df[l])
    
    actual_windGen = pd.concat(a, axis=1)
    
    #add a column with the sum of each column
    actual_windGen['actual_total_Gen'] = actual_windGen.sum(axis=1)
    
    writer = ExcelWriter(r'C:\Users\daniel.mcglynn\Documents\WindForecast\Forecast_Results\Generator_Forecast_Results\final_result\Results.xlsx', engine='openpyxl')
    forecast_windGen.to_excel(writer, sheet_name = 'Forecast Wind Gen')
    writer.save()
    
    #Actual prices sheet
    actual_windGen.to_excel(writer, sheet_name = 'Actual Wind Gen')
    writer.save()
    
    
    pd.concat([forecast_windGen['forecast_total_Gen'], actual_windGen['actual_total_Gen']], axis=1).to_excel(writer, sheet_name = 'Total Wind Gen')
    writer.save()
 