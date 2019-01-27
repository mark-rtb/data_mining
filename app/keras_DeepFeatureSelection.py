"""
This module is designed to analyze signs DeepFutureSelection method
 Receives the directory data format .csv divides them into training and
 test component, normalizes the data. 
 Builds a fully connected multi-input neural network. 
 and trains her to predict the target trait. 
 after training, the weights of neurons in the input circuits are taken modulo,
 sorted and displayed as a result of the analysis.

@author: марк
"""
import pandas as pd
import os
from keras.models import Model
from keras.layers import Dense, Input, Add # the two types of neural network layer we will be using
from sklearn.model_selection import train_test_split

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

    df = df.dropna()

    return df

def data_set_create(df, binary_attribute):

    X = df.iloc[:,1:].values
    y = df[binary_attribute].values

    from sklearn.preprocessing import StandardScaler 
    scale_features_std = StandardScaler() 
    features_train = scale_features_std.fit_transform(X) 

    # Feature scaling with MinMaxScaler 
    from sklearn.preprocessing import MinMaxScaler 
    scale_features_mm = MinMaxScaler() 
    features_train = scale_features_mm.fit_transform(features_train) 

    X_train, X_test, y_train, y_test = train_test_split(features_train, y, test_size=0.2, random_state=55)

    return X_train, X_test, y_train, y_test


def neural_network_create(X_train, input_features):

    def input_DFS():
        input_model = Input(shape=(1,))
        return input_model
    
    def Dence_lauer(i, inp):
        hiden = Dense(1, activation='relu')(inp[i])
        return hiden
    
    inp =[]
    for input_lauer in range(X_train.shape[1]):
        inp.append(input_DFS())
    
    dence = []   
    for i in range(len(inp)):
        dence.append(Dence_lauer(i, inp))
        
    merge = Add()(dence)
    hid = Dense(input_features, activation='relu')(merge)
    b = Dense(1, activation='sigmoid')(hid)
        
    model = Model(inputs=inp, outputs=b)
       
    model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy
    return model

def model_fit(model, X_train, X_test, y_train, y_test, batch_size, num_epochs):

    
    list_Xtrain = []
    for input_data in range(X_train.shape[1]):
        list_Xtrain.append(X_train[:,input_data])
        
    model.fit(list_Xtrain, y_train, # Train the model using the training set...
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=1, validation_split=0.2) # ...holding out 10% of the data for validation
    
    return model

def calc_deepFS(model, X_train, df):

    x = model.get_weights()
    list_weight = []
    for value in range(X_train.shape[1]*2):
        if value % 2 == 0:
            list_weight.append(x[value])
            
    list_name = list(df.columns.values)[1:]
    
    list_abs_weight = []
    for value in list_weight:
        value = abs(value)
        list_abs_weight.append(value)
        
    DeepSelect = pd.DataFrame({'name':list_name, 'value':list_abs_weight})

    return DeepSelect

def main():
    
    list_drop_name = ['ORDERID', 'APPROVED', 'ISSUED', 'SHTRAFDAYSQUANT', 'ORDERSTATUS']
#   list of columns not involved in this stage of analysis

    binary_attribute = 'BAD'
#   target attribute by which the analysis is performed

    key_bin_words_0, key_bin_words_1 = 'не вернул', 'вернул'
#   words that need to be replaced with a binary sign
    
    file_name = 'loanform_features.csv'
    
    dir_name = 'D:\\DeepFeatureSelection\\keras_app\\'
    
    batch_size = 32 
    num_epochs = 300 
    
    df_load = read_data(dir_name, file_name)
    
    df = preprocessing_df(list_drop_name, binary_attribute, key_bin_words_0,
                      key_bin_words_1, df_load)
    
    X_train, X_test, y_train, y_test = data_set_create(df, binary_attribute)
    
    input_features = X_train.shape[1]
    
    model = neural_network_create(X_train, input_features)
    
    model = model_fit(model, X_train, X_test, y_train, y_test, batch_size, num_epochs)
    
    df_DFS = calc_deepFS(model, X_train, df)

    print(df_DFS.sort_values(by='value', ascending=False))
    
    
if __name__ == '__main__':
    main()
