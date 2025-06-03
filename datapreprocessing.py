import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_pre_processing():
    #Dataset reading and dataframe creation
    df_geracao = pd.read_csv('geracao_energia.csv')

    #MIN/MAX Normalization
    scaler = MinMaxScaler()
    df_geracao['val_geracao_norm'] = scaler.fit_transform(df_geracao[['val_geracao']])

    #Dataset train/test spliting 21.8 % for test
    df_train = df_geracao['val_geracao_norm'][:1500].to_numpy()
    df_test = df_geracao['val_geracao_norm'][1500:].to_numpy()
    return df_train,df_test,df_geracao['val_geracao_norm'],scaler

