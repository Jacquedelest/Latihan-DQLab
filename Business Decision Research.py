# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:41:07 2021

@author: Jacque de l'est
"""

"Import Data & Inspection"
import pandas as pd
df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/data_retail.csv', sep=';')
print('Lima data teratas:')
print(df.head())
print('\nInfo dataset:')
print(df.info())

"Data Cleansing"
#kolom First_Transaction
df['First_Transaction'] = pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
#kolom Last_Transaction
df['Last_Transaction'] = pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')
print('Lima data teratas:')
print(df.head())
print('\nInfo dataset:')
print(df.info())

"Churn Customers"
#pengecekan transaksaksi terakhir dalam dataset
print(max(df['Last_Transaction']))
#klasifikasikan customer yang berstatus churn atau tidak dengan boolean
df.loc[df['Last_Transaction'] <= '2018-08-01','is_churn'] = True 
df.loc[df['Last_Transaction'] > '2018-08-01','is_churn'] = False 
print('Lima data teratas:')
print(df.head())
print('\nInfo dataset:')
print(df.info())

"Hapus Kolom yang Tidak Diperlukan"
del df['no']
del df['Row_Num']
#cetak lima data teratas
print(df.head())

"Customers Acquisition by Year"
import matplotlib.pyplot as plt
#kolom tahun transaksi pertama
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
# Kolom tahun transaksi terakhir
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year
df_year = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
df_year.plot(x='Year_First_Transaction', y='Customer_ID', kind='bar', title='Graph of Customer Acquisition')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Customer')
plt.tight_layout()
plt.show()

"Transaction by Year"
plt.clf()
df_year = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
df_year.plot(x='Year_First_Transaction', y='Count_Transaction', kind='bar', title='Graph of Transaction Customer')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Transaction')
plt.tight_layout()
plt.show()

"Average Transaction Amount by Year"
import seaborn as sns
plt.clf()
sns.pointplot(data = df.groupby(['Product', 'Year_First_Transaction'])
			  .mean()
			  .reset_index(), 
			  x='Year_First_Transaction',
			  y='Average_Transaction_Amount', hue='Product')
plt.tight_layout()
plt.show()

"Proporsi Churned Customer untuk Setiap Produk"
plt.clf()
#melakukan pivot data dengan pivot_table
df_piv = df.pivot_table(index='is_churn', 
                        columns='Product',
                        values='Customer_ID', 
                        aggfunc='count', 
                        fill_value=0)
#mendapatkan proportion Churn by Product
plot_product = df_piv.count().sort_values(ascending=False).head(5).index
# Plot pie chartnya
df_piv = df_piv.reindex(columns=plot_product)
df_piv.plot.pie(subplots=True,
                figsize=(10, 7),
                layout=(-1, 2),
                autopct='%1.0f%%',
                title='Proportion Churn by Product')
plt.tight_layout()
plt.show()

"Distribusi Kategorisasi Count Transaction"
plt.clf()
#kategorisasi jumlah transaksi
def func(row):
    if row['Count_Transaction'] == 1:
        val = '1. 1'
    elif (row['Count_Transaction'] >1 and row['Count_Transaction'] <= 3):
        val ='2.2-3'
    elif (row['Count_Transaction'] >3 and row['Count_Transaction'] <= 6):
        val ='3.4-6'
    elif (row['Count_Transaction'] >6 and row['Count_Transaction'] <= 10):
        val ='4.7-10'
    else:
        val ='5.>10'
    return val
#tambahkan kolom baru
df['Count_Transaction_Group'] = df.apply(func, axis=1)
df_year = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
df_year.plot(x='Count_Transaction_Group', y='Customer_ID', kind='bar', title='Customer Distribution by Count Transaction Group')
plt.xlabel('Count_Transaction_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
plt.show()

"Distribusi Kategorisasi Average Transaction Amount"
plt.clf()
#kategorisasi rata-rata besar transaksi
def f(row):
    if (row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <=200000):
        val ='1. 100.000 - 250.000'
    elif (row['Average_Transaction_Amount'] >250000 and row['Average_Transaction_Amount'] <= 500000):
        val ='2. >250.000 - 500.000'
    elif (row['Average_Transaction_Amount'] >500000 and row['Average_Transaction_Amount'] <= 750000):
        val ='3. >500.000 - 750.000'
    elif (row['Average_Transaction_Amount'] >750000 and row['Average_Transaction_Amount'] <= 1000000):
        val ='4. >750.000 - 1.000.000'
    elif (row['Average_Transaction_Amount'] >1000000 and row['Average_Transaction_Amount'] <= 2500000):
        val ='5. >1.000.000 - 2.500.000'
    elif (row['Average_Transaction_Amount'] >2500000 and row['Average_Transaction_Amount'] <= 5000000):
        val ='6. >2.500.000 - 5.000.000'
    elif (row['Average_Transaction_Amount'] >5000000 and row['Average_Transaction_Amount'] <= 10000000):
        val ='7. >5.000.000 - 10.000.000'
    else:
        val ='8. >10.000.000'
    return val
#tambahkan kolom baru
df['Average_Transaction_Amount_Group'] = df.apply(f, axis=1)
df_year = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
df_year.plot(x='Average_Transaction_Amount_Group', y='Customer_ID',kind='bar', title='Customer Distribution by Average Transaction Amount Group')
plt.xlabel('Average_Transaction_Amount_Group')
plt.ylabel('Num_of_Customer')
plt.tight_layout()
plt.show()


"""
Modelling Dataset
"""

"Feature Columns & Target"
#feature column: Year_Diff
df['Year_Diff']=df['Year_Last_Transaction']-df['Year_First_Transaction']
#nama-nama feature columns
feature_columns = ['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']
#features variable
X = df[feature_columns]
#target variable
y = df['is_churn']

"Split X dan y ke dalam bagian Training dan Testing"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

"Train, Predict, & Evaluate"
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#fit the model with data
LogisticRegression().fit(X_train, y_train)
#predict model
y_pred = LogisticRegression().predict(X_test)
# Evaluasi model menggunakan confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix:\n', cnf_matrix)


"Visualisasi Confusion Matrix"
#import required modules
import numpy as np
plt.clf()
#name  of classes
class_names = [0, 1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

"Visualisasi Confusion Matrix"
# import required modules
plt.clf()
#name  of classes
class_names = [0, 1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

"Accuracy, Precision, & Recall"
from sklearn.metrics import accuracy_score, precision_score, recall_score
#menghitung Accuracy, Precision, dan Recall
print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, average='micro'))
print('Recall   :', recall_score(y_test, y_pred, average='micro'))