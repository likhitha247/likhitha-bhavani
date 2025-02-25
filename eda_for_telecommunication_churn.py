# -*- coding: utf-8 -*-
"""eda for telecommunication churn

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hT3VsxUsj_2SDwDFuHIg_66HBsdlZXJB
"""

#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/Tele-communication Churn.csv')
data

data.info()

#data structure
print(type(data))
print(data.shape)

data.shape

#data types
data.dtypes

#drop  duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"state"], axis=1)
data1

#drop  duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"area.code"], axis=1)
data1

#checking for duplicated row in the table
#print the duplicated row
data1[data1.duplicated(keep = False)]

#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()

#find the mean and median values of each numeric column
median_state = data1['state'].median()
median_area.code = data1['area.code'].median()
median_account.length = data1['account.length'].median()
mean_state= data1['state'].mean()
mean_area.code = data1['area.code'].mean()
mean_account.length = data1['account.length'].mean()
print("Median of state: ",median_state)
print("Median of area.code: ",median_area.code)
print("Median of account.length : ",median_account.length)
print("Mean of state: ",mean_state)
print("Mean of area.code: ",mean_area.code)
print("Mean of account.length : ",mean_account.length)

#visualize data1 missing values
cols = data1.columns
colours = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)

#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2,1,figsize=(8,16), gridspec_kw={'height_ratios':[1,3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data1['state'],ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title('Boxplot')
axes[0].set_xlabel('state')

plt.figure (figsize=(6,3))
plt.title("Box plot for tele_communication")
plt.boxplot(data1["intl.mins"], vert = False)
plt.show()

sns.histplot(data1['intl.mins'], kde = True,stat='density',)
plt.show()