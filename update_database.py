# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:37:19 2020

@author: Daniel.McGlynn
"""

import glob
import pandas as pd
import numpy as np
import pypyodbc as db
from pandas import ExcelWriter
import datetime as dt

#write dataframes to sql database

def get_latest_weather():
    """ This function will put last weeks weather data into a dataframe """
    connection = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=METOffice;'
                                'Trusted_connection=yes')
    
    df_weather = pd.read_sql("""SELECT DISTINCT
                                    created_at,
                                    ThreeHourForecast.SiteLocationID,
                                    ThreeHourForecast.StartTime,
                                    EndTime,
                                    MaxUVIndex,
                                    WeatherType,
                                    Visibility,
                                    Temperature,
                                    WindSpeed,
                                    PrecipitationProbability,
                                    Humidity,
                                    WindGust,
                                    FeelsLikeTemperature,
                                    WindDirection,
                                    ObservationRetrieved
                                    FROM
                                      (SELECT
                                           MAX(ObservationRetrieved) AS created_at,
                                           ThreeHourForecast.SiteLocationID,
                                           ThreeHourForecast.StartTime
                                           FROM
                                        	  ThreeHourForecast
                                           where
                                        	  ThreeHourForecast.StartTime >= (CURRENT_TIMESTAMP - 2)
                                           GROUP BY
                                        	  ThreeHourForecast.SiteLocationID, ThreeHourForecast.StartTime) AS latest_forecast
                                    INNER JOIN
                                      ThreeHourForecast
                                    ON
                                      ThreeHourForecast.ObservationRetrieved = latest_forecast.created_at and 
                                      ThreeHourForecast.SiteLocationID = latest_forecast.SiteLocationID and 
                                      ThreeHourForecast.StartTime = latest_forecast.StartTime
                                    order by SiteLocationID, StartTime""", connection)
                          
    print(df_weather.shape)
    
    connection.close()
    
    return df_weather

def wind_direction_numerical(df_weather):
    """ This function changes wind direction to numerical data """
    
    df_merged = df_weather

    l = [s for s in df_merged.columns if "winddirection" in s]
    
           
    for i in df_merged.index:
        for j in l: 
            if df_merged[j].iloc[i] == 'N':
                df_merged.set_value(i, j+' winddir', '1')
            if df_merged[j].iloc[i] == 'NNE':
               df_merged.set_value(i, j+' winddir', '2')
            if df_merged[j].iloc[i] == 'NE':
               df_merged.set_value(i, j+' winddir', '3')
            if df_merged[j].iloc[i] == 'ENE':
               df_merged.set_value(i, j+' winddir', '4')
            if df_merged[j].iloc[i] == 'E':
               df_merged.set_value(i, j+' winddir', '5')
            if df_merged[j].iloc[i] == 'ESE':
               df_merged.set_value(i, j+' winddir', '6')
            if df_merged[j].iloc[i] == 'SE':
               df_merged.set_value(i, j+' winddir', '7')
            if df_merged[j].iloc[i] == 'SSE':
               df_merged.set_value(i, j+' winddir', '8')
            if df_merged[j].iloc[i] == 'S':
               df_merged.set_value(i, j+' winddir', '9')
            if df_merged[j].iloc[i] == 'SSW':
               df_merged.set_value(i, j+' winddir', '10')
            if df_merged[j].iloc[i] == 'SW':
               df_merged.set_value(i, j+' winddir', '11')
            if df_merged[j].iloc[i] == 'WSW':
               df_merged.set_value(i, j+' winddir', '12')
            if df_merged[j].iloc[i] == 'W':
               df_merged.set_value(i, j+' winddir', '13')
            if df_merged[j].iloc[i] == 'WNW':
               df_merged.set_value(i, j+' winddir', '14')
            if df_merged[j].iloc[i] == 'NW':
               df_merged.set_value(i, j+' winddir', '15')
            if df_merged[j].iloc[i] == 'NNW':
               df_merged.set_value(i, j+' winddir', '16')
               
    df_merged['winddirection winddir'] = df_merged['winddirection winddir'].astype(int)
    
    return df_merged

def resample_weather(df_weather):
    """ This function resamples weather data from 3H to 1H"""
    
    connection = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=METOffice;'
                                'Trusted_connection=yes')
    
    df_sl = pd.read_sql('select * FROM SiteLocation', connection)
    
    #get list of closest weather stations to CE Turbines from windMachineLearning database
    conn_clst_stn = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=windMachineLearning;'
                                'Trusted_connection=yes')
    
    df_clst_stn = pd.read_sql('select weather_station_id FROM generators', conn_clst_stn)

    df_sl = df_sl[['sitelocationid','elevation', 'latitude', 'longtitude']]
    
    df = pd.merge(df_weather, df_sl, left_on='sitelocationid', right_on = 'sitelocationid', how='left')
    
    l = []
    
    for i in df_clst_stn['weather_station_id'].astype(int).unique():
       df_1 = df[(df['starttime'] > dt.datetime(2019,11,1)) & (df['sitelocationid'] == i)] 
       
       df_1.drop_duplicates(subset=['starttime', 'sitelocationid'], inplace=True, keep='last')
       
       df_1.index = df_1['starttime']
       
       #split dataframe into columns to interpolate and columns to forward fill
       df_interpolate = pd.DataFrame(index = df_1.index)
       df_ffill = pd.DataFrame(index = df_1.index)
       
       #columns to interpolate
       df_interpolate = df_1[['windspeed', 'windgust', 'winddirection winddir']]
       
       #columns to forward fill
       df_ffill = df_1[['sitelocationid']]
       
       #resample dataframes from 3H to 1H
       df_interpolate = df_interpolate.resample('1H').interpolate()
       
       df_ffill = df_ffill.resample('1H').ffill()
       
       #merge both dataframes
       df_1 = pd.merge(df_ffill, df_interpolate, left_index = True, right_index=True, how='left') 
       
       #merge both dataframes together
       
       #add siteLocationID to df_1 columns names
       #df_1.columns = [str(col) + '_'+str(i) for col in df_1.columns]
       
       l.append(df_1)
       
    #concat all dataframes
    df_1 = pd.concat(l, axis=0, sort=False)
    
    
    return df_1

def get_metered_generation():
    connection = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=MMInBound;'
                                'Trusted_connection=yes')
    
    #df = pd.read_sql('select * FROM MeteredGenerationInfo598', connection)
    
    df_meteredGen = pd.read_sql('select * FROM MeteredGenerationInfo598 WHERE IntervalPeriodTimeStamp between ? and ?', 
                     connection, params=(str(dt.datetime.now()-dt.timedelta(days=2))[:-3], str(dt.datetime.now())[:-3]),)
    
    df_header = pd.read_sql('select * FROM MMHeader598', connection)
    
    connection.close()
    
    return df_header, df_meteredGen

#read each turbine and it's closest weather station (windMachinelearning generators table) 
def get_generator_ID():
    """   """
    #df_CETurbines_ClosestWeatherStnData = pd.read_excel('CETurbines_ClosestWeatherStn.xlsx')
    
    connection = db.connect('Driver={SQL Server};'
                            'Server=CLKSQL001\SQL2014;'
                            'Database=windMachineLearning;'
                            'Trusted_connection=yes')
    
    df_CETurbines_ClosestWeatherStnData = pd.read_sql('select * FROM generators', connection)

    cursor = connection.cursor()
    
    cursor.close()
    connection.close()
    
    return df_CETurbines_ClosestWeatherStnData


def interpolate_weather_data(df_header, df_meteredGen, df_CETurbines_ClosestWeatherStnData):
    """ This function interpolates weather data   """
    
    dataframes = []
    
    for click_generatorID in df_CETurbines_ClosestWeatherStnData['generator_id'].unique():
        
        print(click_generatorID)
    
        #df_turbines = pd.read_excel('CETurbines_ClosestWeatherStn.xlsx')
        df_turbines = df_CETurbines_ClosestWeatherStnData
        
        #filter out Turbines that are outside the area of interest
        #df_turbines = df_turbines[df_turbines['Reference_point_dist'] <= cutoff_dist]
        
        #Ensure all in 'click_generatorID is a string
        for i in range(0,df_turbines.shape[0]):
            df_turbines.set_value(df_turbines.index[i],'generator_id', str(df_turbines['generator_id'].iloc[i]))
        
        #filter closest turbines to weather station of interest. 
        df_turbines = df_turbines[df_turbines['generator_id'].isin([str(click_generatorID)])]
        
        df_merged = pd.merge(df_meteredGen, df_header, left_on='mmheaderid', right_on = 'marketmessageheaderid', how='left')
        
        df_merged1 = df_merged[['meteredgenerationid', 'mmheaderid', 'intervalperiodtimestamp',
               'settlementinterval', 'generationunitmeteredgeneration',
               'lossadjustedgenerationunitmeteredgeneration', 'generationunitid',
               'settlementrunindicator']]
        
        df_CETurbines = df_merged1[df_merged1['generationunitid'].isin(list(set(df_turbines['generator_id']) & set(df_merged1['generationunitid'].unique())))]
        
        """Filter 2nd ReAggregation at M+13 settlement runs """
        df_CETurbines10 = df_CETurbines[df_CETurbines['settlementrunindicator'] == 20]
        
        df_CETurbines10 = df_CETurbines10.groupby(['intervalperiodtimestamp']).sum()
        
        #df_CETurbines10.index = df_CETurbines10['intervalperiodtimestamp']
        
        df_CETurbines10 = df_CETurbines10.resample('1H').sum()
        
        #Use DataFile2_distToRef to find Weather station site location ids to use in ThreeHourForecastresampled
        
        #df_weatherStnDist = pd.read_excel(r'C:\Users\daniel.mcglynn\Documents\WindForecast\DataFile2_distToRef.xlsx')
        
        #filter out waether data from waether stations that are outside the area of interst
        #df_weatherStnDist = df_weatherStnDist[df_weatherStnDist['Reference_point_dist'] <= cutoff_dist]
        
        #select colunmns that have same SiteLocationID as df_weatherStnDist
        #df_weatherData = pd.read_excel(r'C:\Users\daniel.mcglynn\Documents\WindForecast\ThreeHourForecast4_Winddir_resampled.xlsx')
        stn_no = int(df_turbines[df_turbines['generator_id'] == click_generatorID]['weather_station_id'].iloc[0])
        
        df_weatherData = df_weather[df_weather['sitelocationid'] == stn_no]
        
        #create a list l with WindSpeed+'_'+df_weatherStnDist['SiteLocationID'].unique()[i]
        l = []
        #for i in range(0, df_weatherStnDist['SiteLocationID'].unique().shape[0]):
            #l.append('WindSpeed'+'_'+str(df_weatherStnDist['SiteLocationID'].unique()[i]))
            
        #windspeed_colname =  'WindSpeed_'+str(df_CETurbines_ClosestWeatherStnData['weather_station_id'].iloc[df_CETurbines_ClosestWeatherStnData['generator_id'].unique().tolist().index(click_generatorID)])
        #winddir_colname = 'WindDirection winddir_'+str(df_CETurbines_ClosestWeatherStnData['weather_station_id'].iloc[df_CETurbines_ClosestWeatherStnData['generator_id'].unique().tolist().index(click_generatorID)])
        #windgust_colname = 'WindGust_'+str(df_CETurbines_ClosestWeatherStnData['weather_station_id'].iloc[df_CETurbines_ClosestWeatherStnData['generator_id'].unique().tolist().index(click_generatorID)])
        
        windspeed_colname = 'windspeed'
        winddir_colname = 'winddirection winddir'
        windgust_colname = 'windgust'
        
        l.append(windspeed_colname)
        l.append(winddir_colname)
        l.append(windgust_colname)
       
        
        #Use list l to find closest weather station data
        df_weatherData = df_weatherData[l]
        
        df_weatherTurbinesMerged = pd.merge(df_weatherData, df_CETurbines10, left_index = True, right_index = True, how='left')
        
        df_weatherTurbinesMerged['Date'] = df_weatherTurbinesMerged.index
        
        df_weatherTurbinesMerged = df_weatherTurbinesMerged[['Date', 'windspeed', 'winddirection winddir', 'windgust', 'lossadjustedgenerationunitmeteredgeneration']]
        
        df_weatherTurbinesMerged.columns = ['date', 'wind_speed', 'wind_direction', 'wind_gust', 'generation']
        
        
        #convert interpolated wind direction values to integers
        #df_weatherTurbinesMerged[winddir_colname] = df_weatherTurbinesMerged['wind_direction'].astype(int)
        
        #add column to df_weatherTurbinesMerged called 'generator_id' holding generator id
        for i in range(0,df_weatherTurbinesMerged.shape[0]):
            df_weatherTurbinesMerged.set_value(df_weatherTurbinesMerged.index[i], 'generator_id', str(click_generatorID))
        
        #convert generation_id column to type string
        df_weatherTurbinesMerged['generator_id'] = df_weatherTurbinesMerged['generator_id'].astype(str)
        
        #add df_weatherTurbinesMerged to a list
        dataframes.append(df_weatherTurbinesMerged)
    
    #concat all dataframes
    results = pd.concat(dataframes, axis=0, sort=False)
    
    results.fillna(0, inplace=True)
        
    return results


if __name__ == '__main__':
    df_weather = get_latest_weather()
    df_weather = wind_direction_numerical(df_weather)
    df_weather = resample_weather(df_weather)
    
    df_header, df_meteredGen = get_metered_generation()
    df_CETurbines_ClosestWeatherStnData = get_generator_ID()
    df_weatherTurbinesMerged = interpolate_weather_data(df_header, df_meteredGen, df_CETurbines_ClosestWeatherStnData)
    
    
    connection = db.connect('Driver={SQL Server};'
                                'Server=CLKSQL001\SQL2014;'
                                'Database=windMachineLearning;'
                                'Trusted_connection=yes')
    
    cursor = connection.cursor()
    
    """
    for index, row in df_weatherTurbinesMerged.iterrows():
        cursor.execute(IF EXISTS (SELECT generator_id, date from weather where ?,?)  
                          BEGIN
                            UPDATE weather
                              SET wind_speed = values(?), wind_direction = ?, wind_gust = ?, generation = ?
                              WHERE generator_id = ?, date = ?
                          END
                          ELSE
                          BEGIN
                            INSERT INTO weather(generator_id , date, wind_speed, wind_direction, wind_gust, generation)
                            VALUES ?,?,?,?,?,?
                          END , (row['generator_id'], row['date'], 'TEST', row['wind_direction'], row['wind_gust'], 
                          row['generation'], row['generator_id'], row['date'], row['generator_id'], row['date'], row['wind_speed'], 
                          row['wind_direction'], row['wind_gust'], row['generation']))
    """
    
    counter = 0   
    for index, row in df_weatherTurbinesMerged.iterrows():
        cursor.execute("SELECT count(1) from weather where generator_id = ? and date = ?", (row['generator_id'], row['date']))
        no_records = cursor.fetchone()[0]
        if no_records == 1:
            cursor.execute("""UPDATE weather
                              SET [wind_speed] = ?, [wind_direction] = ?, [wind_gust] = ?, [generation] = ?
                              WHERE [generator_id] = ? AND [date] = ?""", (row['wind_speed'], row['wind_direction'], 
                              row['wind_gust'], row['generation'], row['generator_id'], row['date']))
            connection.commit() 
            print(counter / int(df_weatherTurbinesMerged.shape[0]))
        elif no_records == 0:
            cursor.execute("""INSERT INTO weather([generator_id] , [date], [wind_speed], [wind_direction], [wind_gust], [generation])
                            VALUES (?,?,?,?,?,?)""", (row['generator_id'], row['date'], row['wind_speed'], row['wind_direction'], 
                            row['wind_gust'], row['generation']))
            connection.commit() 
            print(counter / int(df_weatherTurbinesMerged.shape[0]))
        counter+=1
                       
    
    """
    #write data to weather table  
    for index, row in df_weatherTurbinesMerged.iterrows():
        cursor.execute("INSERT INTO weather(generator_id, date, wind_speed, wind_gust, wind_direction, generation) values (?,?,?,?,?,?)", 
                       (row[data.columns[0]], row['Date'], row['WindSpeed'], row['WindDirection'], row['WindGust'], row['Price']) )
        connection.commit()   
    """    
    #close connection
    cursor.close()
    connection.close()


