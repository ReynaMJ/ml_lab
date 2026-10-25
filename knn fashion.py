#Question 8
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

#convert pixel to features
X_train=X_train.reshape(len(X_train),-1)
X_test=X_test.reshape(len(X_test),-1)

#normalize pixel values
X_train=X_train/255.0
X_test=X_test/255.0

k_val=[1,3,5,20]

for k in k_val:
  print("K:",k)

  model=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
  model.fit(X_train[:10000],y_train[:10000])

  pred=model.predict(X_test[:2000])

  print("Accuracy score:",accuracy_score(y_test[:2000],pred))
  print("Classification Report:\n",classification_report(y_test[:2000],pred))

