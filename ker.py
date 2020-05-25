from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


dataset=mnist.load_data('mymnist.db')
train,test=dataset
trainx,trainy=train
testx,testy=test
X_train_1d = trainx.reshape(-1 , 28*28)
X_test_1d = testx.reshape(-1 , 28*28)
trainx = X_train_1d.astype('float32')
testx = X_test_1d.astype('float32')
trainycat=to_categorical(trainy)
model=Sequential()
model.add(Dense(units=256,input_dim=784,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy'])
h = model.fit(trainx, trainycat, epochs=1,verbose=0)
print(h.history['accuracy'][-1])

