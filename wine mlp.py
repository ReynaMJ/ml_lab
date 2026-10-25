#question 13
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Input

data=pd.read_csv("WineQT.csv")

X=data.drop('quality',axis=1)
y=data['quality']

le=LabelEncoder()
y=le.fit_transform(y)
y=to_categorical(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

def create_model(layer):
  model=Sequential()

  model.add(Input(shape=(X_train.shape[1],)))

  for neuron in layer:
    model.add(Dense(neuron,activation="relu"))

  model.add(Dense(y.shape[1],activation='softmax'))

  model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

  return model

arch={
    "Model 1 (Simple)": [8],
    "Model 2 (Medium)": [32, 16],
    "Model 3 (Deep)": [64, 32, 16]
}

for name,layer in arch.items():
  print("Model:",name)

  model=create_model(layer)

  model.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=0)

  loss,acc=model.evaluate(X_test,y_test,verbose=0)
  print("Accuracy:",acc)


