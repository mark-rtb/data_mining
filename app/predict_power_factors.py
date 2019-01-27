"""
The program is designed to analyze customer data during credit scoring.
when you call, you must enter the directory with the file of the customer data
@author: mark-rtb
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import core libraries

from Inf_val import data_vars
#import information value (IV) calculation function-measures


def read_data(dir_name, file_name):
    """ The function of reading data for analysis in the format .csv
    On input accepts-the directory in which the file is located and
    the file name with the extension
    The output returns pandas DataFrame read from the file """


    file_name = os.path.join(dir_name, file_name)
#   full path to the data file

    load_dat_df = pd.read_csv(file_name)
#   read data from a file .csv and create a pandas Dataframe

    return load_dat_df


def preprocessing_df(list_drop_name, binary_attribute, key_bin_words_0,
                     key_bin_words_1, df):
    """ the function of data preprocessing
    the input accepts a list of deleted columns, a column name to convert to
    numeric type, keywords to convert, pandas DataFrame
    in the process, prints the processing result as a table
    the output returns pandas Dataframe preprocessed """


    df = df.dropna(subset=[binary_attribute])
#   remove zero values

    df.drop(list_drop_name, axis=1, inplace=True)
#   remove columns with unnecessary information

    df[str(binary_attribute)].replace(key_bin_words_1, 1, inplace=True)
    df[str(binary_attribute)].replace(key_bin_words_0, 0, inplace=True)
#   replace string values in columns with numeric values

    print(df.head())
#   print part of DataFrame

    return df


def vis_signs(df):
    """ The function builds a dependency between pairs of components in
    the transmitted pandas Dataframe """


    correlation = df.corr(method='pearson')
#   Pearson correlation coefficients are calculated

    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.title('Correlation between different fearures')
    plt.show()
#   building a map of signs


def calc_inf_val(df, binary_attribute):
    """ The calculation to measure the predictive power of the variable
    the input accepts a pre-processed pandas DataFame and a sign for calculation
    prints a table
    the output returns a sorted DataFrame """


    IV = data_vars(df, df[binary_attribute])[1]
#   The calculation to measure the predictive power of the variable

    inf_val = IV.sort_values('IV', ascending=False)
#   sorting features from larger to smaller

    print(inf_val)

    return inf_val


def save_inf_val(inf_val, binary_attribute, dir_name):
    """ the function saves the sorted DataFrame into a directory
    with data with the name of the attribute on which the calculation was
    performed, in the format .pkl """


    path_to_file = os.path.join(dir_name, binary_attribute)
#   full path to the data file

    inf_val.to_pickle(path_to_file)
#   saving file


def main():
    """ The function initializes the variables and performs the calculation
    functions in the order necessary to obtain lists of factors affecting
    the probability of loan default and the probability of loan approval """


    dir_name = 'D:\\data_mining\\data\\'
#   the directory where the data file is located

    file_name = 'loanform_features.csv'
#   the name of the data file

    list_drop_name = ['ORDERID', 'APPROVED', 'ISSUED', 'SHTRAFDAYSQUANT',
                      'ORDERSTATUS']
#   list of columns not involved in this stage of analysis

    binary_attribute = 'BAD'
#   target attribute by which the analysis is performed

    key_bin_words_0, key_bin_words_1 = 'не вернул', 'вернул'
#   words that need to be replaced with a binary sign

    df_load = read_data(dir_name, file_name)
    df = preprocessing_df(list_drop_name, binary_attribute, key_bin_words_0,
                          key_bin_words_1, df_load)
    vis_signs(df)
    inf_val = calc_inf_val(df, binary_attribute)
    save_inf_val(inf_val, binary_attribute, dir_name)
#    run the analysis to determine the factors affecting the loan default


    list_drop_name = ['ORDERID', 'BAD', 'ISSUED', 'SHTRAFDAYSQUANT',
                      'ORDERSTATUS']
    binary_attribute = 'APPROVED'
    key_bin_words_0, key_bin_words_1 = 'отказано', 'одобрено'
#   we re-declare variables to change the key factor of the analysis

    df = preprocessing_df(list_drop_name, binary_attribute, key_bin_words_0,
                          key_bin_words_1, df_load)
    vis_signs(df)
    inf_val = calc_inf_val(df, binary_attribute)
    save_inf_val(inf_val, binary_attribute, dir_name)
#   run the analysis to determine the factors that affect the likelihood
#   of a loan approval


if __name__ == '__main__':
    main()
