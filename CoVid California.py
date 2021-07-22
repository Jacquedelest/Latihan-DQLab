# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:04:12 2021

@author: Jacque de l'est
"""


import pandas as pd
df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")
df['month'] = pd.to_datetime(df['date'], format="%d.%m.%y").dt.month_name() #membuat kolom bulan dari kolom date
df.set_index("date", inplace=True) #menjadikan date sebagai index
df.drop('state', axis=1, inplace=True) #menghapus kolom state karena hanya ada satu state
print(df.head())
df.info() #mengecek info data
df.describe() #menampilkan info summary untuk kolom numerik
print(df['month'].value_counts()) #grouping nilai month
print(df.groupby('month')['cases'].sum()) #total infeksi per tiap bulan
print(df['cases'].sum()) #total jumlah kasus