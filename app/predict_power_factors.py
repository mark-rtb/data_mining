# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:44:44 2019

@author: марк
"""

import pandas as pd
import os
import seaborn as sns

DATA_DIR = 'D:\\credit_scoring\\data\\'
file_name = os.path.join(DATA_DIR, "loanform_features.csv") 
load_dat_df = pd.read_csv(file_name)
corr_matrix = load_dat_df.drop(['APPROVED', 'ISSUED'], axis=1).corr()

sns.heatmap(corr_matrix);
