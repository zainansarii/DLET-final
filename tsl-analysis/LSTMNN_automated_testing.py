import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime as dt
import numpy as np
import pandas as pd      
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# initialise arrays filled with training/testing dates over an 8 week window

#demand indexing
start_demand_train_date = dt.date(2019,6,1)
end_demand_train_date = dt.date(2019,12,31)
start_demand_test_date = dt.date(2019,6,1)
end_demand_test_date = dt.date(2020,1,7)
#niv indexing
start_niv_train_date = dt.date(2019,6,1)
end_niv_train_date = dt.date(2019,12,31)
start_niv_test_date = dt.date(2020,1,1)
end_niv_test_date = dt.date(2020,1,7)
#trade indexing
start_trade_date = dt.date(2020,1,1)
end_trade_date = dt.date(2020,1,7)
#set delta
delta = dt.timedelta(days=7)

start_demand_train_arr = []
end_demand_train_arr = []
start_demand_test_arr = []
end_demand_test_arr = []
start_niv_train_arr = []
end_niv_train_arr = []
start_niv_test_arr = []
end_niv_test_arr = []
start_trade_arr = []
end_trade_arr = []

for i in range(8):
    s1 = start_demand_train_date.strftime("%y%m%d") + "01"
    start_demand_train_arr.append(int(s1))

    s2 = end_demand_train_date.strftime("%y%m%d") + "48"
    end_demand_train_arr.append(int(s2))

    s3 = start_demand_test_date.strftime("%y%m%d") + "01"
    start_demand_test_arr.append(int(s3))

    s4 = end_demand_test_date.strftime("%y%m%d") + "48"
    end_demand_test_arr.append(int(s4))

    s5 = start_niv_train_date.strftime("%y%m%d") + "02"
    start_niv_train_arr.append(int(s5))

    s6 = end_niv_train_date.strftime("%y%m%d") + "48"
    end_niv_train_arr.append(int(s6))

    s7 = start_niv_test_date.strftime("%y%m%d") + "01"
    start_niv_test_arr.append(int(s7))

    s8 = end_niv_test_date.strftime("%y%m%d") + "47"
    end_niv_test_arr.append(int(s8))

    s9 = start_trade_date.strftime("%y%m%d") + "04"
    start_trade_arr.append(int(s9))

    s10 = end_trade_date.strftime("%y%m%d") + "44"
    end_trade_arr.append(int(s10))

    start_demand_train_date = start_demand_train_date + delta
    end_demand_train_date = end_demand_train_date + delta 
    start_demand_test_date = start_demand_test_date + delta
    end_demand_test_date = end_demand_test_date + delta
    start_niv_train_date = start_niv_train_date + delta
    end_niv_train_date = end_niv_train_date + delta
    start_niv_test_date = start_niv_test_date + delta
    end_niv_test_date = end_niv_test_date + delta
    start_trade_date = start_trade_date + delta
    end_trade_date = end_trade_date + delta

print(start_demand_train_arr)
print(end_demand_train_arr)
print(start_demand_test_arr)
print(end_demand_test_arr)
print(start_niv_train_arr)
print(end_niv_train_arr)
print(start_niv_test_arr)
print(end_niv_test_arr)
print(start_trade_arr)
print(end_trade_arr)

for z in range(8):

    # read in demand data with missing values imputed
    df_demand = pd.read_csv('demand_data.csv')
    
    # select data between relevent time indexes from full demand dataset
    startindextrain = df_demand[df_demand['id'] == start_demand_train_arr[z]].index.tolist()[0]
    endindextrain = df_demand[df_demand['id'] ==  end_demand_train_arr[z]].index.tolist()[0]
    startindextest = df_demand[df_demand['id'] == start_demand_test_arr[z]].index.tolist()[0]
    endindextest = df_demand[df_demand['id'] == end_demand_test_arr[z]].index.tolist()[0]

    # split demand dataframe into training and testing
    df_demand = df_demand.drop(['id'], axis=1)
    train_dataset = df_demand.loc[startindextrain:endindextrain]
    test_dataset = df_demand.loc[startindextest:endindextest]

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('rsd')
    test_labels = test_features.pop('rsd')

    # reshape demand input data
    n_future = 1
    n_past = 4

    trainX = []
    trainY = []
    testX = []
    testY = []

    for i in range(n_past, len(train_dataset) - n_future + 1):
      trainX.append(train_features[i - n_past:i])
      trainY.append(train_labels[i + n_future - 2:i + n_future -1])
    trainX, trainY = np.array(trainX), np.array(trainY)

    for i in range(n_past, len(test_dataset) - n_future + 1):
      testX.append(test_features[i - n_past:i])
      testY.append(test_labels[i + n_future - 2:i + n_future - 1])
    testX, testY = np.array(testX), np.array(testY)

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(trainX))

    # build, compile, and train demand model    
    model = keras.Sequential()
    model.add(normalizer)
    model.add(layers.LSTM(32, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(layers.Dense(trainY.shape[1]))
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))

    history = model.fit(
        trainX, trainY,
        validation_split=0.2, epochs=200, verbose=0)
    
    print("Demand loss: " + model.evaluate(testX, testY, verbose=0))
    test_predictions = model.predict(testX).flatten().tolist()

    # read in NIV and trading data with missing values imputed
    df_niv = pd.read_csv('niv_data.csv')
    df_trade = df_niv[['indexpricet+2','imbalancepricet+2']]

    # select data between relevent time indexes from full NIV dataset
    startindextrain = df_niv[df_niv['id'] == start_niv_train_arr[z]].index.tolist()[0]
    endindextrain = df_niv[df_niv['id'] == end_niv_train_arr[z]].index.tolist()[0]
    startindextest = df_niv[df_niv['id'] == start_niv_test_arr[z]].index.tolist()[0]
    endindextest = df_niv[df_niv['id'] == end_niv_test_arr[z]].index.tolist()[0]
    startindextrade = df_niv[df_niv['id'] == start_trade_arr[z]].index.tolist()[0]
    endindextrade = df_niv[df_niv['id'] == end_trade_arr[z]].index.tolist()[0]
    df_niv = df_niv.drop(['id','indexpricet+2','imbalancepricet+2'], axis=1)

    # insert demand predictions into niv dataframe
    niv_dataset = df_niv.loc[startindextrain:endindextest]
    niv_dataset['predicteddemand'] = test_predictions
    test_predictions = test_predictions[2:]
    for i in range(2):
      test_predictions.append(0)
    niv_dataset['predicteddemandt+2'] = test_predictions

    # split niv dataframe into training and testing
    train_dataset = niv_dataset.loc[startindextrain:endindextrain]
    test_dataset = niv_dataset.loc[startindextest:endindextest]
    test_dataset.drop(test_dataset.tail(2).index,inplace=True)

    df_trade = df_trade.loc[startindextrade:endindextrade]

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('nivoutput')
    test_labels = test_features.pop('nivoutput')

    # reshape NIV input data
    n_future = 1
    n_past = 2

    trainX = []
    trainY = []
    testX = []
    testY = []

    for i in range(n_past, len(train_dataset) - n_future + 1):
      trainX.append(train_features[i - n_past:i])
      trainY.append(train_labels[i + n_future - 2:i + n_future -1])
    trainX, trainY = np.array(trainX), np.array(trainY)

    for i in range(n_past, len(test_dataset) - n_future + 1):
      testX.append(test_features[i - n_past:i])
      testY.append(test_labels[i + n_future - 2:i + n_future - 1])
    testX, testY = np.array(testX), np.array(testY)

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(trainX))

    # build, compile, and train NIV model    
    model = keras.Sequential()
    model.add(normalizer)
    model.add(layers.LSTM(32, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(trainY.shape[1]))
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))

    history = model.fit(
        trainX, trainY,
        validation_split=0.2, epochs=20, verbose=0)
    
    # make NIV predictions and feed into trading dataframe
    print("NIV loss: " + model.evaluate(testX, testY, verbose=0))
    test_predictions = model.predict(testX).flatten().tolist()
    df_trade['nivt+2'] = test_predictions
    df_trade = df_trade.dropna()

    niv = df_trade['nivt+2'].to_list()
    mip = df_trade['indexpricet+2'].to_list()
    imp = df_trade['imbalancepricet+2'].to_list()

    # execute trading algorithm and print accuracy
    good_trades = 0
    bad_trades = 0

    for i in range(len(niv)):
        if niv[i] > 0:
            if mip[i] <= imp[i]:
                good_trades+=1
            else:
                bad_trades+=1
        else:
            if mip[i] >= imp[i]:
                good_trades+=1
            else:
                bad_trades+=1

    total_trades = good_trades+bad_trades
    accuracy = (good_trades/total_trades)*100
    print(accuracy)
