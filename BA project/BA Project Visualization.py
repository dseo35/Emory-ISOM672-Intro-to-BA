
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
import numpy as np

#%%
data = pd.read_csv("C:/Users/10331/OneDrive/Documents/GitHub/Emory-ISOM672-Intro-to-BA/BA project/hotel_bookings.csv")
data.shape
data.describe()
data.dtypes
data.head()
is_string_dtype(data["hotel"])

#%% Feature Selection 1
df = data[:]
df.drop("arrival_date_year",axis =1, inplace = True)
df.drop("reservation_status",axis =1, inplace = True)
df.drop("reservation_status_date",axis =1, inplace = True)
df.drop("company",axis =1, inplace = True)
df.drop("assigned_room_type",axis =1, inplace = True)

df.dtypes

#%%
for i in range(len(df. columns)):
    if is_string_dtype(df.iloc[:,i]):
        sns.countplot(df.iloc[:,i])
        plt.show
    else:
        sns.distplot(df.iloc[:,i],kde=False)
        plt.show()
        if sum(df.iloc[:,i]>(df.iloc[:,i].quantile(0.75)*2.5 - df.iloc[:,i].quantile(0.25)*1.5)) > 0:
            sns.distplot(df.iloc[:,i],kde=False)
            plt.ylim(0,10)
            plt.xlim(df.iloc[:,i].quantile(0.75),df.iloc[:,i].max()+5)
            plt.show()

#%%
num = list(df. columns)
for i in list(df. columns):
    for j in num:
        if is_string_dtype(df.loc[:,i]):
            sns.catplot(i,j,data = df)
            plt.show
        elif is_string_dtype(df.loc[:,j]):
            sns.catplot(j,i,data = df)
            plt.show
        else:
            sns.relplot(i,j,data = df)
            plt.show()
    del num[0]

#%%
for i in list(df. columns):
    sns.catplot("is_canceled",i,data = df)
    plt.show()

#%%
viz1 = data.loc[:,["agent","arrival_date_year"]]
viz2 = viz1.groupby(["agent","arrival_date_year"]).size()
viz2 = viz2.reset_index()
viz2.dtypes
tmp = viz2[viz2.arrival_date_year == 2017]
tmp.groupby(0).size()
tmp = tmp[tmp[0] > 100]
viz2 = viz2[viz2.agent.isin(tmp.agent)]

tmp = viz2.groupby(["agent"]).sum()
tmp = tmp[tmp[0] > 500]
tmp = viz2[viz2.0 > ]

#%%

#Heatmap CM
corr= df.corr()
mask = np.triu(np.ones_like(corr))


plt.figure(figsize=(16, 16))
ax = sns.heatmap(
    corr,
    vmin=-.5, vmax=.5, center=0,
    cmap= sns.diverging_palette(20,220, n=200),
    square = True, 
    linewidth=4,
    annot = True, 
    fmt='.1g',
    mask = mask

    )

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment = 'right'
    )

#%%

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

categories = ['customer_type', 'market_segment']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)

X, Y = df.iloc[:,2:], df.is_canceled
cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')
X_vec = cv.fit_transform(X)

res = dict(zip(cv.get_feature_names(),
               mutual_info_classif(X_vec, Y, discrete_features=True)
               ))
print(res)