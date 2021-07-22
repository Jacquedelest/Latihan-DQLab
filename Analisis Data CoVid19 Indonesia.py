# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:56:47 2021

@author: Jacque de l'est
"""

"Akses API CoVid19.co.id"
import requests
resp = requests.get('https://data.covid19.go.id/public/api/update.json')
print(resp.headers) #cek metadata yang tersimpan

"Ekstrak Isi Respon"
cov_id_raw = resp.json()
print('Length of cov_id_raw : %d.' %len(cov_id_raw))
print('Komponen cov_id_raw  : %s.' %cov_id_raw.keys())
cov_id_update = cov_id_raw['update']

"Analisa Data"
print('Tanggal pembaharuan data penambahan kasus :', cov_id_update['penambahan']['tanggal'])
print('Jumlah penambahan kasus sembuh :', cov_id_update['penambahan']['jumlah_sembuh'])
print('Jumlah penambahan kasus meninggal :', cov_id_update['penambahan']['jumlah_meninggal'])
print('Jumlah total kasus positif hingga saat ini :', cov_id_update['total']['jumlah_positif'])
print('Jumlah total kasus meninggal hingga saat ini:', cov_id_update['total']['jumlah_meninggal'])

"CoVid-19 di NTT"
import requests
resp_ntt = requests.get('https://data.covid19.go.id/public/api/prov_detail_NUSA_TENGGARA_TIMUR.json')
cov_ntt_raw = resp_ntt.json()

"Memahami Kasus CoVid-19 di NTT"
print('Nama-nama elemen utama:\n', cov_ntt_raw.keys())
print('\nJumlah total kasus COVID-19 di Nusa Tenggara Timur : %d' %cov_ntt_raw['kasus_total'])
print('Persentase kematian akibat COVID-19 di Nusa Tenggara Timur : %f.2%%' %cov_ntt_raw['meninggal_persen'])
print('Persentase tingkat kesembuhan dari COVID-19 di Nusa Tenggara Timur : %f.2%%' %cov_ntt_raw['sembuh_persen'])

"Memperoleh Informasi Secara Lengkap"
import numpy as np
import pandas as pd
cov_ntt = pd.DataFrame(cov_ntt_raw['list_perkembangan'])
print('Info CoVid di NTT:\n', cov_ntt.info())
print('\nLima data teratas CoVid di NTT:\n', cov_ntt.head())

"Menjinakan Data"
cov_ntt_tidy = (cov_ntt.drop(columns=[item for item in cov_ntt.columns if item.startswith('AKUMULASI') or item.startswith('DIRAWAT')]).rename(columns=str.lower).rename(columns={'kasus': 'kasus_baru'})
)
cov_ntt_tidy['tanggal'] = pd.to_datetime(cov_ntt_tidy['tanggal']*1e6, unit='ns')
print('Lima data teratas:\n', cov_ntt_tidy.head())


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.clf()

"Grafik Kasus Positif"
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(data=cov_ntt_tidy, x='tanggal', height='kasus_baru', color="salmon")
fig.suptitle('Kasus Harian Positif COVID-19 di NTT', 
             y=1.00, fontsize=16, fontweight='bold', ha='center')
ax.set_title('Terjadi pelonjakan kasus di awal bulan Februari',
             fontsize=10)
ax.set_xlabel('')
ax.set_ylabel('Jumlah kasus')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id', color='blue',
        ha='right', transform=ax.transAxes)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.grid(axis='y')
plt.tight_layout()
plt.show()

"Grafik Kasus Sembuh"
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(data=cov_ntt_tidy, x='tanggal', height='sembuh', color='olivedrab')
ax.set_title('Kasus Harian Sembuh Dari COVID-19 di NTT',
             fontsize=22)
ax.set_xlabel('')
ax.set_ylabel('Jumlah kasus')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id', color='blue',
        ha='right', transform=ax.transAxes)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.grid(axis='y')
plt.tight_layout()
plt.show()

"Grafik Kasus Meninggal"
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(data=cov_ntt_tidy, x='tanggal', height='meninggal', color='slategrey')
ax.set_title('Kasus Harian Meninggal Dari COVID-19 di NTT',
             fontsize=22)
ax.set_xlabel('')
ax.set_ylabel('Jumlah kasus')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id', color='blue',
        ha='right', transform=ax.transAxes)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.grid(axis='y')
plt.tight_layout()
plt.show()

"Data per Minggu"
cov_ntt_pekanan = (cov_ntt_tidy.set_index('tanggal')['kasus_baru']
                                   .resample('W')
                                   .sum()
                                   .reset_index()
                                   .rename(columns={'kasus_baru': 'jumlah'})
                    )
cov_ntt_pekanan['tahun'] = cov_ntt_pekanan['tanggal'].apply(lambda x: x.year)
cov_ntt_pekanan['pekan_ke'] = cov_ntt_pekanan['tanggal'].apply(lambda x: x.weekofyear)
cov_ntt_pekanan = cov_ntt_pekanan[['tahun', 'pekan_ke', 'jumlah']]

print('Info CoVid NTT Mingguan:')
print(cov_ntt_pekanan.info())
print('\nLima data teratas CoVid NTT Mingguan:\n', cov_ntt_pekanan.head())
cov_ntt_pekanan['jumlah_pekanlalu'] = cov_ntt_pekanan['jumlah'].shift().replace(np.nan, 0).astype(np.int)
cov_ntt_pekanan['lebih_baik'] = cov_ntt_pekanan['jumlah'] < cov_ntt_pekanan['jumlah_pekanlalu']

print('Sepuluh data teratas:\n', cov_ntt_pekanan.head(10))

"Bar Chart Mingguan"
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(data=cov_ntt_pekanan, x='pekan_ke', height='jumlah',
	   color=['mediumseagreen' if x is True else 'salmon' for x in cov_ntt_pekanan['lebih_baik']])
fig.suptitle('Kasus Pekanan Positif COVID-19 di NTT',
			 y=1.00, fontsize=16, fontweight='bold', ha='center')
ax.set_title('Kolom hijau menunjukan penambahan kasus baru lebih sedikit dibandingkan satu pekan sebelumnya', fontsize=12)
ax.set_xlabel('')
ax.set_ylabel('Jumlah kasus')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id',
		color='blue', ha='right', transform=ax.transAxes)

plt.grid(axis='y')
plt.tight_layout()
plt.show()

"Kasus Aktif Saat Ini"
cov_ntt_akumulasi = cov_ntt_tidy[['tanggal']].copy()
cov_ntt_akumulasi['akumulasi_aktif'] = (cov_ntt_tidy['kasus_baru'] - cov_ntt_tidy['sembuh'] - cov_ntt_tidy['meninggal']).cumsum()
cov_ntt_akumulasi['akumulasi_sembuh'] = cov_ntt_tidy['sembuh'].cumsum()
cov_ntt_akumulasi['akumulasi_meninggal'] = cov_ntt_tidy['meninggal'].cumsum()
cov_ntt_akumulasi.tail()

"Line Chart"
fig, ax = plt.subplots(figsize=(10,5))
ax.plot('tanggal', 'akumulasi_aktif', data=cov_ntt_akumulasi, lw=2)

ax.set_title('Akumulasi Aktif CoVid-19 di NTT',
             fontsize=22)
ax.set_xlabel('')
ax.set_ylabel('Akumulasi Aktif')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id', color='blue',
        ha='right', transform=ax.transAxes)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.grid()
plt.tight_layout()
plt.show()

"Kasus Aktif, Sembuh, dan Meninggal"
fig, ax = plt.subplots(figsize=(10,5))
cov_ntt_akumulasi.plot(x='tanggal', kind='line', ax=ax, lw=3,
						color=['salmon','slategrey','olivedrab'])
ax.set_title('Dinamika Kasus Covid-19 di Nusa Tenggara Timur', fontsize=22)
ax.set_xlabel('')
ax.set_ylabel('Akumulasi Aktif')
ax.text(1, -0.1, 'Sumber data: covid.19.go.id', color='blue',
	   ha='right', transform=ax.transAxes)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.grid()
plt.tight_layout()
plt.show()