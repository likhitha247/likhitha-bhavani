#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


# Printing the information 
data.info()


# In[4]:


# Dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[5]:


# Drop dupplicate column( Temp C) and Unnamed column

# data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1, inplace = True)
data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1.info()


# In[7]:


# Convert the Month column data type to float data type

data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


# Print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[9]:


# Checking for duplicated rows in the table
#Print only the duplicated row (one) only
data1[data1.duplicated()]


# In[10]:


# Drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# #### Rename the columns

# In[12]:


# Change column names (Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# #### Impute the missing values

# In[14]:


# Display data1 info()
data1.info()


# In[15]:


# Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[16]:


# Visualize data1 missing values using heat map

cols = data1.columns 
colors = ['black', 'yellow'] 
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[17]:


# Find the mean and median values of each numeric column
#Imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[18]:


# Replace the Ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[19]:


# Find the mean and median values of each numeric column
#Imputation of missing value with median
median_Solar = data1["Solar"].median()
mean_Solar = data1["Solar"].mean()
print("Median of Solar: ", median_Solar)
print("Mean of Ozone: ", mean_Solar)


# In[20]:


# Replace the Solar missing values with mean value
data1['Solar'] = data1['Solar'].fillna(mean_Solar)
data1.isnull().sum()


# In[21]:


# print the data1 5 rows
data1.head()


# In[22]:


# Find the mode values of categorical column (weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[23]:


# Impute missing values (Replace NaN with  mode etc.) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[24]:


# Impute missing values (Replace NaN with  mode etc.) of "month" using fillna()
mode_month = data1["Month"].mode()[0]
data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[25]:


data1.tail()


# In[26]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[27]:


data1["weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[28]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[29]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[30]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

sns.histplot(data["Ozone"], kde=True, ax=axes[0], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency") 
                         
plt.tight_layout()
plt.show()           


# In[31]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin plot")


# In[32]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[33]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert= False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[34]:


data1["Ozone"].describe()


# In[35]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x> (mu + 3*sigma))):
        print(x)


# In[36]:


import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# ## observations from Q-Q plot
# - The data does not follow normal distribution as the data points are deviating significantly awaya from the red line
# - The data shows a right-skewed distribution and possible outliers

# In[79]:


sns.violinplot(data=data1["Ozone"], color='grey')
plt.title("Violin Plot")
plt.show


# In[87]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="Orange",palette="Set2",size=6)


# In[93]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange",size=6, jitter = True)


# In[89]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="red")


# In[91]:


sns.boxplot(data=data1, x = "Weather", y="Ozone")


# In[95]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[99]:


data1["Wind"].corr(data1["Temp"])


# ##it is a mild negative corelation

# In[ ]:




