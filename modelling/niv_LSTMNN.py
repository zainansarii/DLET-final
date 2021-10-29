import numpy as np
import pandas as pd      
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# import data and drop ID column
df_train = pd.read_csv('niv_data.csv')
df_train = df_train.drop(['id', 'indexprice', 'imbalanceprice'], axis=1)
df_train.head(5)

# check for missing values and impute with mean of the series
df_train.isnull().sum()
df_train = df_train.fillna(df_train.mean())

# split training and testing data 80:20 chronologically
train_dataset = df_train.head(int(len(df_train)*(0.8)))
test_dataset = df_train.tail(int(len(df_train)*(0.2)))
df_train.describe().transpose()

# split features from labels in both training and testing datasets
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('nivoutput')
test_labels = test_features.pop('nivoutput')

# reshape input data using sliding window algorithm
trainX = []
trainY = []
testX = []
testY = []
future_pred = 1
timesteps = 2

for i in range(timesteps, len(train_dataset) - future_pred + 1):
  trainX.append(train_features[i - timesteps:i])
  trainY.append(train_labels[i + future_pred - 2:i + future_pred -1])
trainX, trainY = np.array(trainX), np.array(trainY)

for i in range(timesteps, len(test_dataset) - future_pred + 1):
  testX.append(test_features[i - timesteps:i])
  testY.append(test_labels[i + future_pred - 2:i + future_pred - 1])
testX, testY = np.array(testX), np.array(testY)

# instantiate normaliser and adapt to training data
normaliser = preprocessing.Normalization(axis=-1)
normaliser.adapt(np.array(trainX))

intial = np.array(trainX[0][0])
with np.printoptions(precision=2, suppress=True):
  print('First example:', intial)
  print('Normalized:', normaliser(intial).numpy())
  
# define function to plot training loss
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 15000])
  plt.xlabel('Epoch')
  plt.ylabel('Error [output]')
  plt.legend()
  plt.grid(True)
  
# build and compile neural network
model = keras.Sequential()
model.add(normaliser)
model.add(layers.LSTM(32, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(layers.LSTM(64, activation='relu', return_sequences=True))
model.add(layers.LSTM(128, activation='relu', return_sequences=True))
model.add(layers.LSTM(64, activation='relu', return_sequences=True))
model.add(layers.LSTM(32, activation='relu', return_sequences=False))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(trainY.shape[1]))

model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
model.summary()

# train model and plot loss
history = model.fit(
    trainX, trainY,
    validation_split=0.2, epochs=20)
plot_loss(history)

# evaluate model performance on test data
performance = model.evaluate(testX, testY, verbose=0)
print('Mean absolute error [niv] = ' + str(performance))

# make predictions to be used in trading algorithm
test_predictions = model.predict(testX).flatten()
df_demand = pd.DataFrame({'demand':test_predictions.tolist()})
df_demand.to_csv("niv_predictions.csv", index=False)
