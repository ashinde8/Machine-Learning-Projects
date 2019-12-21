#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

res_pur = pd.read_csv('res_purchase_2014.csv')
res['Amount'].shape

len(res['Amount'])

a = res['Amount'][1]

#How much was spend at WW GRAINGER?

grainger = res[res['Vendor']=='WW GRAINGER']['Amount'].astype(float).sum()
grainger


# How much was spend at WM SUPERCENTER?

super = res[res['Vendor']=='WM SUPERCENTER']['Amount'].astype(float).sum()
super


# How much was spend at GROCERY STORES?

grocery = pur[pur['Merchant Category Code (MCC)']=='GROCERY STORES,AND SUPERMARKETS'] 
gro_amt = grocery['Amount'].astype(float).sum()
gro_amt


import re

res_pur_['Amount'] = res_pur_['Amount'].str.replace('^[^\d]*', '').astype(float)

res_pur_['Amount'] = res_pur_['Amount'].str.extract('(\d+\.?\d*)', expand=False)

res_pur_['Amount'].sum()

# Read ’Energy.xlsx’ and ’EnergyRating.xlsx’ as BalanceSheet and Ratings(dataframe).

BalanceSheet = pd.read_excel("Energy.xlsx")

Ratings = pd.read_excel("EnergyRating.xlsx")

BalanceSheet.head()

Ratings.head()

Ratings.columns

BalanceSheet.shape

Ratings.shape

BalanceSheet.info()

#drop the column if more than 90% value in this colnmn is 0. (Hint, using
#pd.apply() function)

Ratings.info()

Ratings.columns

Ratings.drop(['S&P Subordinated Debt Rating'],inplace = True, axis=1)

Ratings.info()

BalanceSheet.head()

#BalanceSheet
BalanceSheet.shape

BalanceSheet.drop(BalanceSheet.columns[BalanceSheet.apply(lambda col: col.isnull().sum() > 759)], axis=1, inplace = True)

BalanceSheet.shape

# replace all None or NaN with average value of each column.

BalanceSheet = BalanceSheet.fillna(BalanceSheet.mean())

Ratings['S&P Domestic Long Term Issuer Credit Rating'].value_counts()

Ratings['S&P Domestic Short Term Issuer Credit Rating'].value_counts()

Ratings['S&P Domestic Long Term Issuer Credit Rating'].fillna("BBB",inplace=True)


Ratings['S&P Domestic Short Term Issuer Credit Rating'].fillna("A-2",inplace=True)


Ratings.isnull().sum()

BalanceSheet.isnull().sum()

BalanceSheet.columns


#Calculate the correlation matrix for variables = [’Current Assets - Other - Total’, ’Current Assets - Total’, ’Other Long-term Assets’, ’Assets Netting Other
#Adjustments’].

data = BalanceSheet[['Current Assets - Other - Total','Current Assets - Total','Other Long-term Assets','Assets Netting & Other Adjustments']]
#d = BalanceSheet['Current Assets - Other - Total']

data.corr()

#If you look at column (’Company Name’), you will find some company name
#end with ’CORP’, ’CO’ or ’INC’. Create a new column (Name: ’CO’) to store
#the last word of company name. (For example: ’CORP’ or, ’CO’ or ’INC’) (Hint:
#using map function)

BalanceSheet['Company Name'].str.endswith('CORP').value_counts()

BalanceSheet['Company Name'].str.endswith('INC').value_counts()

BalanceSheet['Company Name'].str.endswith('CO').value_counts()

companies=['LLC','CORP:CORP','CO','PLC','LTD']
CO=[]
Comp_names=BalanceSheet['Company Name']
for name in Comp_names:
    for a in companies:
        if a in name:
            CO.append(name)
        else:
            CO.append('NaN')

print(CO)
BalanceSheet['CO'] = BalanceSheet['Company Name'].map(lambda x: x, companies)
BalanceSheet['CO']=BalanceSheet['Company Name'].str.split().str[-1]
print(BalanceSheet['CO'])

#Merge (inner) Ratings and BalanceSheet based on ’datadate’ and ’Global Company Key’, and name merged dataset ’Matched’.

Matched = pd.merge(Ratings, BalanceSheet, how ='inner' ,on =["Data Date","Global Company Key"])

Matched.shape

Matched.head()

#Mapping
#For dataset ’Matched’, we have following mapping:

Matched['S&P Domestic Long Term Issuer Credit Rating'].value_counts()

dict = {'AAA' : 0,
'AA+' : 1,
'AA' : 2,
'AA-' : 3,
'A+' : 4,
'A' : 5,
'A-' : 6,
'BBB+' : 7,
'BBB' : 8,
'BBB-' : 9,
'BB+' : 10,
'BB' : 11,
        
}

dict

Matched['Rate']=Matched['S&P Domestic Long Term Issuer Credit Rating'].map(dict)

Matched['Rate']

Fre = Matched['Company Name'].str.contains("CO")

Fre.value_counts()

Matched.to_csv("HW4.csv")

hw4 = pd.read_csv("HW4.csv")

hw4.head()


