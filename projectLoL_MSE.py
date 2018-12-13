from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# generate seed
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# load dataset
df = pd.read_csv('./dataset/editedGames.csv', header=None)
dataset = df.values
X = dataset[:,0:57]
Y = dataset[:,57]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# set models
model = Sequential()
model.add(Dense(100, input_dim=57, activation='relu'))
model.add(Dense(57, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error',
             optimizer='adam',
             metrics=['accuracy'])

# set directory to save models
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# set condition of saving models
modelpath="./model/{epoch:02d}-{val_loss:.4f}.h5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# set automatically stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

# execution model
history = model.fit(X, Y, validation_split=0.2, epochs=2000, batch_size=1000, verbose= 0, callbacks = [early_stopping_callback, checkpointer])

# 예측 값과 실제 값의 비교
# mean_squared_error 이용할 때
Y_prediction = model.predict(X_test).flatten()

for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제 결과: {:.4f}, 예상 결과: {:.4f}".format(label, prediction))

# create empty list to save results
prediction_t1 = []
prediction_t2 = []


# push results
for i in range(round(len(df)*0.2)):
    label = Y_test[i]

    if(Y_test[i] == 1):
        prediction_t1.append(Y_prediction[i])
    if(Y_test[i] == 2):
        prediction_t2.append(Y_prediction[i])

# draw output
plt.title("Output")
plt.xlabel("predicted team")
plt.ylabel("number of samples")
plt.hist(prediction_t1, color="red", bins=200, range=[0.8, 2.2], label='team 1')
plt.hist(prediction_t2, color="blue", bins=200, range=[0.8, 2.2], label='team 2')
plt.legend(loc="upper center")
plt.show()
plt.clf()

# save the validation loss to y_vloss
y_vloss=history.history['val_loss']

# save the validation accuracy to y_vacc
y_vacc=history.history['val_acc']

# save the train loss to y_loss
y_loss=history.history['loss']

# save the train accuracy to y_acc
y_acc=history.history['acc']

# set x value. show the loss by color RED and the accuracy by color BLUE
x_len = np.arange(len(y_vacc))
plt.title("Accuracy and Loss for Train, Validation")
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Validation loss')
plt.plot(x_len, y_vacc, "o", c="blue", markersize=2, label= 'Validation accuracy')
plt.plot(x_len, y_loss, "-", c="m", markersize=1, label='Train loss')
plt.plot(x_len, y_acc, "-", c="c", markersize=1, label='Train accuracy')

plt.legend(loc='upper left')

plt.show()