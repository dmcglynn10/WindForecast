# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:07:48 2019

@author: Daniel.McGlynn
"""
import tkinter as tk
from tkinter import *
import sys
import os
import datetime as dt
sys.path.insert(0, r'C:\Users\daniel.mcglynn\Documents\WindForecast\windForecast_GUI')
import windMachineLearning_readDatabase
import update_database
#import print_number
sys.path.insert(0, 'ISEM_datafiles/ISEM_DAM_results/')
#import concat_DAM_results_xl
#import concat_SEMO_DAM_results_xl
sys.path.insert(0, 'C:/Users/daniel.mcglynn/Documents/Eirgrid Data/')
#import concat_Wind_Demand_Data
sys.path.insert(0, 'C:/Users/daniel.mcglynn/Documents/Eirgrid Data/')
#import concat_Wind_Demand_Data_IBM
sys.path.insert(0, 'C:/Users/daniel.mcglynn/Documents/Eirgrid Data/')
#import TestingSqlServer_ISEM
#import TestingSqlServer_ISEM_IMB
#import hourlyModelISEM
#import hourlyModelISEM_IMB
#import plot_results_weekdays_weekends_gui
#import plot_results_weekdays_weekends_IMB_gui


def openProgram():
    print('Update database')
    print('Started')
    sys.path.insert(0, 'C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI')
    os.system('python update_database.py ')
    #print_number.print_number()
    #concat_DAM_results_xl.concat_DAM_results_xl()
    #concat_SEMO_DAM_results_xl.concat_DAM_results_xl()
    #TestingSqlServer_ISEM.read_data_fromdb()
    #concat_Wind_Demand_Data.concat_Wind_Demand_Data()
    #concat_Wind_Demand_Data.concat_forecast_wind()
    #concat_Wind_Demand_Data.concat_Forecast_Demand()
    #concat_Wind_Demand_Data.remove_nan_datafile()
    print('Finished')

def update_database_1H():
    print('Update database')
    print('Started')
    sys.path.insert(0, 'C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI')
    os.system('C:\WINDOWS\system32\cmd.exe /K python update_database.py ')
    #TestingSqlServer_ISEM_IMB.read_data_fromdb()
    #concat_Wind_Demand_Data_IBM.concat_Wind_Demand_Data_IMB()
    #concat_Wind_Demand_Data_IBM.concat_forecast_wind_IMB()
    #concat_Wind_Demand_Data_IBM.concat_forecast_Demand_IMB()
    #concat_Wind_Demand_Data_IBM.remove_nan_datafile_IMB()
    print('finished')
    
def update_database_30min():
    print('Update database')
    print('Started')
    sys.path.insert(0, 'C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI')
    os.system('C:\WINDOWS\system32\cmd.exe /K python update_database_30min.py ')
    #TestingSqlServer_ISEM_IMB.read_data_fromdb()
    #concat_Wind_Demand_Data_IBM.concat_Wind_Demand_Data_IMB()
    #concat_Wind_Demand_Data_IBM.concat_forecast_wind_IMB()
    #concat_Wind_Demand_Data_IBM.concat_forecast_Demand_IMB()
    #concat_Wind_Demand_Data_IBM.remove_nan_datafile_IMB()
    print('finished')
    
def IMBForecast(v_IMB, UL_IMB, LL_IMB, No_days_IMB, E_day_IMB):
    print('IMBForecast')
    print('Started')
    print(int(v_IMB.get()))
    print(int(UL_IMB.get()))
    print(int(LL_IMB.get()))
    print(str(No_days_IMB.get()))
    print(str(E_day_IMB.get()))
    #os.system('python hourlyModelISEM_IMB.py '+str(v_IMB.get())+' '+str(UL_IMB.get())+' '+str(LL_IMB.get())+' '+str(No_days_IMB.get())+' '+str(E_day_IMB.get()))
    os.system('python plot_results_weekdays_weekends_IMB_gui.py')
    print('finished')
    
def DAMForecast(regressor_select, No_days, E_day, mode):
    print('DAMForecast')
    print(regressor_select.get())
    print(No_days.get())
    print(E_day.get())
    print(type(E_day.get()))
    print(mode.get())
    sys.path.insert(0, 'C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI')
    os.system('C:\WINDOWS\system32\cmd.exe /K python windMachineLearning_readDatabase.py '+str(regressor_select.get())+' '+str(No_days.get())+' '+E_day.get()+' '+str(mode.get()))#+' '+str(UL.get()))' '
    #os.system('python plot_results_weekdays_weekends_gui.py')
    #os.system('python test.py')
    print('DAM Forecast completed')
    
def WindGen_30min_Forecast(regressor_select, No_days, E_day, mode):
    print('DAMForecast')
    print(regressor_select.get())
    print(No_days.get())
    print(E_day.get())
    print(type(E_day.get()))
    print(mode.get())
    sys.path.insert(0, 'C:\\Users\\daniel.mcglynn\\Documents\\WindForecast\\windForecast_GUI')
    os.system('C:\WINDOWS\system32\cmd.exe /K python windMachineLearning_readDatabase_30min.py '+str(regressor_select.get())+' '+str(No_days.get())+' '+E_day.get()+' '+str(mode.get()))#+' '+str(UL.get()))' '
    #os.system('python plot_results_weekdays_weekends_gui.py')
    #os.system('python test.py')
    print('DAM Forecast completed')
    
def Strategy():
    print('Strategy started')
    os.system('python results_gui.py')
    print('Strategy completed')
    
def DataFile_IMB_():
    
    return
    
    
class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #Setup Menu
        MainMenu(self)
        #setup Frame
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        for F in (StartPage, PageOne, PageTwo):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame(StartPage)
    def show_frame(self, context):
        frame = self.frames[context]
        frame.tkraise()
        
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background="bisque")
        
        leftFrame = tk.Frame(self, background="bisque", width=150, height=250, relief='raised', bd=4)
        leftFrame.grid(column=0, row=0)
        
        #Upp_frame = tk.Frame(leftFrame, background="bisque")
        #Upp_frame.grid(row=0,column=0)
        
        
        label = tk.Label(leftFrame , text="Total CE Wind Generation", bg='bisque')
        label.grid(column=0, row=0, columnspan=5, rowspan=10, padx = 2, pady=10, sticky='nsew')
        
        #Upper limit price label
        #Upp_lab_text = tk.StringVar()
        #Upp_lab_text.set("Upper limit price: ")
        #Upper_label = tk.Label(leftFrame, textvariable=Upp_lab_text, width=15, bg='bisque')
        #Upper_label.grid(column=0, row=10, padx = 2, pady=10)
        
        #Upper limit price entry
        #s = tk.StringVar()
        #UL = tk.Entry(leftFrame, textvariable=s)
        #s.set('120') #default price
        #UL.grid(column=1, row=10, pady=10)
        
        #No_days_frame = tk.Frame(leftFrame, background="bisque")
        #No_days_frame.grid(column=0, row=0)
        
        #No of days to forecast label
        No_days_text = tk.StringVar()
        No_days_text.set("No days to forecast: ")
        No_days_label = tk.Label(leftFrame, textvariable=No_days_text, width=15, bg='bisque')
        No_days_label.grid(column=0, row=11, padx = 2, pady=10, sticky='w')
        
        #No of days to forecast entry
        n = tk.StringVar()
        No_days = tk.Entry(leftFrame, textvariable=n)
        n.set('2') #default price
        No_days.grid(column=1, row=11, padx = 2, pady=10)
        
        #E_day_frame = tk.Frame(leftFrame, background="bisque")
        #E_day_frame.grid(column=2, row=0, sticky='w')
        
        #Today's date plus 2 label (End date)
        E_day_text = tk.StringVar()
        E_day_text.set("End date of forecast: ")
        E_day_label = tk.Label(leftFrame, textvariable=E_day_text, width=15, bg='bisque')
        E_day_label.grid(column=0, row=12, padx = 2, pady=10)
        
        #Today's date plus 2 entry (End date)
        n = tk.StringVar()
        n.set((dt.datetime.now().replace(hour=23,minute=0, second=0, microsecond=0)+dt.timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')) #default price
        E_day = tk.Entry(leftFrame, textvariable=n)
        E_day.grid(column=1, row=12, padx = 2, pady=10)
        
        rightFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        rightFrame.grid(row=0,column=1, sticky='wens', padx=20)
        
        label = tk.Label(rightFrame , text="Model select", bg='bisque')
        label.grid(column=0, row=0, columnspan=5, rowspan=10, padx = 2, pady=10, sticky='nsew')
        
        regressor_select = tk.IntVar(None, 2) 
        tk.Radiobutton(rightFrame , text='AdaBoost', bg='bisque', variable=regressor_select, value=1, command=lambda:print(regressor_select.get())).grid(column=30, row=0, padx=10, sticky='w')
        tk.Radiobutton(rightFrame , text='MLP', bg='bisque', variable=regressor_select, value=2, command=lambda:print(regressor_select.get())).grid(column=30, row=1, padx=10, sticky='w')
        tk.Radiobutton(rightFrame , text='XGBoost', bg='bisque', variable=regressor_select, value=7, command=lambda:print(regressor_select.get())).grid(column=30, row=3, padx=10, sticky='w')
        tk.Radiobutton(rightFrame , text='dNN', bg='bisque', variable=regressor_select, value=8, command=lambda:print(regressor_select.get())).grid(column=30, row=4, padx=10, sticky='w')
        
        modeFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        modeFrame.grid(row=1,column=1, sticky='wens', padx=20, pady=20)
        
        label = tk.Label(modeFrame , text="Mode select", bg='bisque')
        label.grid(column=0, row=0, columnspan=5, rowspan=10, padx = 2, pady=10, sticky='nsew')
        
        mode = tk.IntVar(None, 1) 
        tk.Radiobutton(modeFrame , text='Training', bg='bisque', variable=mode, value=0, command=lambda:print(mode.get())).grid(column=30, row=4, padx=10, sticky='w')
        tk.Radiobutton(modeFrame , text='Predict', bg='bisque', variable=mode, value=1, command=lambda:print(mode.get())).grid(column=30, row=5, padx=10, sticky='w')
        
        
        update_database_1HbtnFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        update_database_1HbtnFrame.grid(row=2,column=0, sticky='wens', padx=20, pady=20)
        
        windGenbtnFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        windGenbtnFrame.grid(row=2,column=1, sticky='wens', padx=20, pady=20)
       

        
        #30min wind forecast button
        windGen30minbtnFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        windGen30minbtnFrame.grid(row=5,column=1, sticky='wens', padx=20, pady=20)
        
        update_database_30minbtnFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        update_database_30minbtnFrame.grid(row=5,column=0, sticky='wens', padx=20, pady=20)
        
        update_database_1Hbtn = tk.Button(update_database_1HbtnFrame , text="Update weather data 1H", command=update_database_1H)
        update_database_1Hbtn.grid(column=30, row=37, pady=10)
        
        DAM_Forecast = tk.Button(windGenbtnFrame , text="Wind Gen Forecast", command=lambda:DAMForecast(regressor_select, No_days, E_day, mode))
        DAM_Forecast.grid(column=30, row=37, sticky='nesw', padx=2, pady=10)
        
        windGen30_Forecast = tk.Button(windGen30minbtnFrame , text="Wind Gen 30min Forecast", command=lambda:WindGen_30min_Forecast(regressor_select, No_days, E_day, mode))
        windGen30_Forecast.grid(column=30, row=37, sticky='nesw', padx=2, pady=10)
        #page_one = tk.Button(rightFrame , text="Page One", command=lambda:controller.show_frame(PageOne))
        #page_one.grid(column=30, row=37, sticky='nesw', padx=2, pady=10)
        #page_two = tk.Button(rightFrame , text="Page Two", command=lambda:controller.show_frame(PageTwo))
        #page_two.grid(column=30, row=38, sticky='nesw', padx=2, pady=10)
        
        
        update_database_30minbtn = tk.Button(update_database_30minbtnFrame, text="update weather data 30min", command=update_database_30min)
        update_database_30minbtn.grid(column=30, row=37, sticky='nesw', padx=2, pady=10)
       
class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background="bisque")
        
        leftFrame = tk.Frame(self, background="bisque", width=150, height=150, relief='raised', bd=4)
        leftFrame.grid(column=0, row=0)
        
        label = tk.Label(leftFrame , text="Page One", bg='bisque')
        label.grid(column=0, row=0, columnspan=5, rowspan=10, pady=10, sticky='nsew')
        
        #Upper limit price label
        Upp_lab_text = tk.StringVar()
        Upp_lab_text.set("Upper limit price: ")
        Upper_label = tk.Label(leftFrame, textvariable=Upp_lab_text, width=15, bg='bisque')
        Upper_label.grid(column=0, row=10, pady=10)
        
        #Upper limit price entry
        s = tk.StringVar()
        UL_IMB = tk.Entry(leftFrame, textvariable=s)
        s.set('500') #default price
        UL_IMB.grid(column=1, row=10, pady=10)
        
        #Lower limit price label
        Low_lab_text = tk.StringVar()
        Low_lab_text.set("Lower limit price: ")
        Lower_label = tk.Label(leftFrame, textvariable=Low_lab_text, width=15, bg='bisque')
        Lower_label.grid(column=0, row=11, pady=10)
        
        #Lower limit price entry
        l = tk.StringVar()
        LL_IMB = tk.Entry(leftFrame, textvariable=l)
        l.set('-500') #default price
        LL_IMB.grid(column=1, row=11, pady=10)
        
        #No_days_frame = tk.Frame(leftFrame, background="bisque")
        #No_days_frame.grid(column=0, row=0)
        
        #No of days to forecast label
        No_days_text = tk.StringVar()
        No_days_text.set("No days to forecast: ")
        No_days_label = tk.Label(leftFrame, textvariable=No_days_text, width=15, bg='bisque')
        No_days_label.grid(column=0, row=12, pady=10, sticky='w')
        
        #No of days to forecast entry
        n = tk.StringVar()
        No_days_IMB = tk.Entry(leftFrame, textvariable=n)
        n.set('7') #default price
        No_days_IMB.grid(column=1, row=12, pady=10)
        
        #E_day_frame = tk.Frame(leftFrame, background="bisque")
        #E_day_frame.grid(column=2, row=0, sticky='w')
        
        #Today's date plus 2 label (End date)
        E_day_text = tk.StringVar()
        E_day_text.set("End date of forecast: ")
        E_day_label = tk.Label(leftFrame, textvariable=E_day_text, width=15, bg='bisque')
        E_day_label.grid(column=0, row=13, pady=10)
        
        #Today's date plus 2 entry (End date)
        n = tk.StringVar()
        E_day_IMB = tk.Entry(leftFrame, textvariable=n)
        n.set((dt.datetime.now()+dt.timedelta(days=3)).strftime('%Y-%m-%d')) #default price
        E_day_IMB.grid(column=1, row=13, pady=10)
        
        rightFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        rightFrame.grid(row=0,column=1, sticky='wens', padx=20)
        
        v_IMB = tk.IntVar(None, 2) 
        tk.Radiobutton(rightFrame , text='AdaBoost', bg='bisque', variable=v_IMB, value=1, command=lambda:print(v_IMB.get())).grid(column=30, row=0, padx=10, sticky='w')
        tk.Radiobutton(rightFrame , text='MLP', bg='bisque', variable=v_IMB, value=2, command=lambda:print(v_IMB.get())).grid(column=30, row=1, padx=10, sticky='w')
        tk.Radiobutton(rightFrame , text='GRNN', bg='bisque', variable=v_IMB, value=6, command=lambda:print(v_IMB.get())).grid(column=30, row=2, padx=10, sticky='w')
        
        DataFile_IMB = tk.Button(rightFrame, text="DataFile_IMB", command=DataFile_IMB_)
        DataFile_IMB.grid(column=30, row=36, sticky='nesw', padx=2, pady=10)
        DataFile_IMB = tk.Button(rightFrame, text="IMB Forecast", command=lambda:IMBForecast(v_IMB, UL_IMB, LL_IMB, No_days_IMB, E_day_IMB))
        DataFile_IMB.grid(column=30, row=37, sticky='nesw', padx=2, pady=10)
        start_page = tk.Button(rightFrame, text="Start Page", command=lambda:controller.show_frame(StartPage))
        start_page.grid(column=30, row=38, sticky='nesw', padx=2, pady=10)
        page_two = tk.Button(rightFrame, text="Page Two", command=lambda:controller.show_frame(PageTwo))
        page_two.grid(column=30, row=39, sticky='nesw', pady=10)
        
        
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background="bisque")
        
        leftFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        leftFrame.grid(column=0, row=0)
        
        label = tk.Label(leftFrame, text="Results Page", bg='bisque')
        label.grid(padx=100, pady=150)
        
        rightFrame = tk.Frame(self, background="bisque", relief='raised', bd=4)
        rightFrame.grid(row=0,column=1, sticky='wens', padx=20)
        
        results_btn = tk.Button(rightFrame, text="Strategy", command=Strategy)
        results_btn.grid(column=30, row=0, sticky='nesw', padx=2, pady=10)
        start_page = tk.Button(rightFrame, text="Start Page", command=lambda:controller.show_frame(StartPage))
        start_page.grid(column=30, row=1, sticky='nesw', padx=2, pady=10)
        page_one = tk.Button(rightFrame, text="Page One", command=lambda:controller.show_frame(PageOne))
        page_one.grid(column=30, row=2, sticky='nesw', padx=2, pady=10)
        
        
class MainMenu:
    def __init__(self, master):
        menubar = tk.Menu(master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=master.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        master.config(menu=menubar)
     
app = App()  
app.geometry("500x500")
app.title("Wind Generation Forecast")
app.mainloop()