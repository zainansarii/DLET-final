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
# in this case, TSL = 3 months
one = dt.date(2019,10,1)
two = dt.date(2019,12,31)
three = dt.date(2020,1,7)
four = dt.date(2020,1,1)
delta = dt.timedelta(days=7)

start_demand_train = [19100101]
end_demand_train = [19123148]
start_demand_test = [19100101]
end_demand_test = [20010748]
start_niv_train = [19100101]
end_niv_train = [19123148]
start_niv_test = [20010101]
end_niv_test = [20010748]

for i in range(7):
    one = one + delta
    two = two + delta
    three = three + delta
    four = four + delta

    s1 = one.strftime("%y%m%d") + "01"
    start_demand_train.append(int(s1))
    start_demand_test.append(int(s1))
    start_niv_train.append(int(s1))

    s2 = two.strftime("%y%m%d") + "48"
    end_demand_train.append(int(s2))
    end_niv_train.append(int(s2))

    s3 = three.strftime("%y%m%d") + "48"
    end_demand_test.append(int(s3))
    end_niv_test.append(int(s3))

    s4 = four.strftime("%y%m%d") + "01"
    start_niv_test.append(int(s4))

print(start_demand_train)
print(end_demand_train)
print(start_demand_test)
print(end_demand_test)
print(start_niv_train)
print(end_niv_train)
print(start_niv_test)
print(end_niv_test)


for z in range(8):
  
  # read in demand data with missing values removed
  df_demand = pd.read_csv('demand_data.csv')

  # select data between relevent time indexes from full demand dataset
  startindextrain = df_demand[df_demand['id'] == start_demand_train[z]].index.tolist()[0]
  endindextrain = df_demand[df_demand['id'] == end_demand_train[z]].index.tolist()[0]
  startindextest = df_demand[df_demand['id'] == start_demand_test[z]].index.tolist()[0]
  endindextest = df_demand[df_demand['id'] == end_demand_test[z]].index.tolist()[0]

  # split demand dataframe into training and testing
  df_demand = df_demand.drop(['id'], axis=1)
  train_dataset = df_demand.loc[startindextrain:endindextrain]
  test_dataset = df_demand.loc[startindextest:endindextest]

  train_features = train_dataset.copy()
  test_features = test_dataset.copy()
  train_labels = train_features.pop('rsd')
  test_labels = test_features.pop('rsd')

  normalizer = preprocessing.Normalization(axis=-1)
  normalizer.adapt(np.array(train_features))

  def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

  # build, compile, and train demand model
  dnn_model = build_and_compile_model(normalizer)
  history = dnn_model.fit(
      train_features, train_labels,
      validation_split=0.2, epochs=100, verbose=0)

  print("Demand loss: " + str(dnn_model.evaluate(test_features, test_labels, verbose=0)))
  test_predictions = dnn_model.predict(test_features).flatten().tolist()

  # read in NIV data (with missing values removed) and trading data
  df_niv = pd.read_csv('niv_data.csv')
  df_trade = df_niv[['indexpricet+2','imbalancepricet+2']]

  # select data between relevent time indexes from full NIV dataset
  startindextrain = df_niv[df_niv['id'] == start_niv_train[z]].index.tolist()[0]
  endindextrain = df_niv[df_niv['id'] == end_niv_train[z]].index.tolist()[0]
  startindextest = df_niv[df_niv['id'] == start_niv_test[z]].index.tolist()[0]
  endindextest = df_niv[df_niv['id'] == end_niv_test[z]].index.tolist()[0]
  df_niv = df_niv.drop(['id','indexpricet+2','imbalancepricet+2'], axis=1)

  #insert demand predictions into niv dataframe
  niv_dataset = df_niv.loc[startindextrain:endindextest]
  niv_dataset['predicteddemand'] = test_predictions
  test_predictions = test_predictions[2:]
  for i in range(2):
    test_predictions.append(0)
  niv_dataset['predicteddemandt+2'] = test_predictions

  # split niv dataframe into training and testing
  train_dataset = df_niv.loc[startindextrain:endindextrain]
  train_dataset = train_dataset.dropna()
  test_dataset = df_niv.loc[startindextest:endindextest]
  test_dataset.drop(test_dataset.tail(2).index,inplace=True)

  
  df_trade = df_trade.loc[startindextest:endindextest]
  df_trade.drop(df_trade.tail(2).index,inplace=True)

  train_features = train_dataset.copy()
  test_features = test_dataset.copy()

  train_labels = train_features.pop('nivoutput')
  test_labels = test_features.pop('nivoutput')

  normalizer = preprocessing.Normalization(axis=-1)
  normalizer.adapt(np.array(train_features))
  
  # build, compile, and train NIV model
  dnn_model = build_and_compile_model(normalizer)
  history = dnn_model.fit(
      train_features, train_labels,
      validation_split=0.2, epochs=50, verbose=0)

  # make NIV predictions and feed into trading dataframe
  print("NIV loss: " + str(dnn_model.evaluate(test_features, test_labels, verbose=0)))
  test_predictions = dnn_model.predict(test_features).flatten().tolist()
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
