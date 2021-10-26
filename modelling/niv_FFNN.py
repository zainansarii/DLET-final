import numpy as np
import pandas as pd      
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# import data and drop columns
df_train = pd.read_csv('niv_data.csv')
df_train = df_train.drop(['id', 'indexprice', 'imbalanceprice'], axis=1)
df_train.head(5)

# check for missing values and drop rows containing NaNs
df_train.isnull().sum()
df_train = df_train.dropna()

# split training and testing data 80:20 chronologically
train_dataset = df_train.head(int(len(df_train)*(0.8)))
test_dataset = df_train.tail(int(len(df_train)*(0.2)))
df_train.describe().transpose()

# split features from labels in both training and testing datasets
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('nivoutput')
test_labels = test_features.pop('nivoutput')

# instantiate normaliser and adapt to training data
normaliser = preprocessing.Normalization(axis=-1)
normaliser.adapt(np.array(train_features))

initial = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
  print('Non-normalised', initial)
  print('Normalised:', normaliser(initial).numpy())
  
# define function to plot training loss
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 300])
  plt.xlabel('Epoch')
  plt.ylabel('Error [output]')
  plt.legend()
  plt.grid(True)
  
# define function to build and compile neural network
def build_and_compile_model(normaliser):
  model = keras.Sequential([
      normaliser,
      layers.Dense(50, activation='relu'),
      layers.Dense(50, activation='relu'),
      layers.Dense(1)
  ])
  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# build and compile neural network
dnn_model = build_and_compile_model(normaliser)
dnn_model.summary()

# train model and plot loss
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.15, epochs=50)
plot_loss(history)

# evaluate model performance on test data
performance = dnn_model.evaluate(test_features, test_labels, verbose=0)
print('Mean absolute error [niv] = ' + performance)

# make predictions to be used in trading algorithm
test_predictions = dnn_model.predict(test_features).flatten()
test_predictions_list = test_predictions.tolist()
df_predicted_niv = pd.DataFrame({'nivt+2':test_predictions_list})
df_predicted_niv.to_csv("niv_predictions.csv", index=False)
