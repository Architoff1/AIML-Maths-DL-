from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=5,input_dim=7,kernel_initializer='normal',activation='relu'))

