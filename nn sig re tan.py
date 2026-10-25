#Question 14
import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.datasets import fashion_mnist

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train[:5000]
y_train=y_train[:5000]
X_test=X_test[:1000]
y_test=y_test[:1000]

X_train=X_train/255.0
X_test=X_test/255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def create_model(act):

  model=Sequential([
      Input(shape=(28,28),),
      Flatten(),
      Dense(128,activation=act),
      Dense(64,activation=act),
      Dense(10,activation='softmax')
  ])

  model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])

  return model

activation=['sigmoid','tanh','relu']


histories={}
results={}
for act in activation:
  print(f"Training model {act}")

  model=create_model(act)
  start=time.time()

  history=model.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=1)

  end=time.time()

  loss,acc=model.evaluate(X_test,y_test,verbose=0)
  results[act]={'accuracy':acc,'time':end-start}

  histories[act]=history

for act in results:
  print(f"For {act} activation model:\nAccuracy:{results[act][accuracy]},Time:{results[act][time]}")
