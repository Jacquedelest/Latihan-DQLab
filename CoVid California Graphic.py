# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:29:17 2021

@author: Jacque de l'est
"""


import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")
df.drop('state', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'], format="%d.%m.%y")
df['month'] = df['date'].dt.month
df.set_index('date', inplace=True)

#df[df['month']==12]['cases'].plot() #line chart
#df[df['month'==12[['cases', 'deaths']].plot()
#df.groupby('month')['cases'].sum().plot(kind="bar") #bar chart
#df.groupby('month')[['cases', 'deaths']].sum().plot(kind="barh", stacked=True) #horizontal bar chart
#df[df["month"]==6]["cases"].plot(kind="box") #box plot 
#df[df["month"]==6]["cases"].plot(kind="hist", bins=10) #histogram
#df[df["month"]==6][["cases", "deaths"]].plot(kind="area", stacked=False) #area plot
#df[df["month"]==6][["cases", "deaths"]].plot(kind="scatter", x='cases', y='deaths') #scatter plot
#df.groupby('month')['cases'].sum().plot(kind="pie") #pie chart
"Plot formatting"
df[df['month']==6][['cases', 'deaths']].plot(kind="line", legend=True, color=['yellow', '#E73E19'])
plt.xlabel('Days in June')
plt.ylabel('Number')
plt.suptitle("COVID-19 in June")
plt.savefig('plot.png')
