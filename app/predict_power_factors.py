# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:44:44 2019

@author: марк
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Inf_val import data_vars

DATA_DIR = 'D:\\data_mining\\data\\'
file_name = os.path.join(DATA_DIR, "loanform_features.csv") 
load_dat_df = pd.read_csv(file_name)

load_dat_df.dropna(inplace= True)

df_info = load_dat_df.head()

load_dat_df.drop([ 'ORDERID', 'APPROVED', 'ISSUED',], axis=1, inplace= True)


load_dat_df['BAD'].replace("вернул",1,inplace= True)
load_dat_df['BAD'].replace("не вернул",0,inplace= True)
df_info = load_dat_df.head()


correlation = load_dat_df.corr(method='spearman')
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')
plt.show()


final_iv, IV = data_vars(load_dat_df, load_dat_df.BAD)
IV.sort_values('IV', ascending=False)
