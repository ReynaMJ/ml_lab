#question 12

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn import svm
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train[:5000]
y_train=y_train[:5000]
X_test=X_test[:1000]
y_test=y_test[:1000]

X_train=X_train.reshape(len(X_train),-1)
X_test=X_test.reshape(len(X_test),-1)

X_train=X_train/255.0
X_test=X_test/255.0

model1=svm.SVC(kernel="linear")
model1.fit(X_train,y_train)
pred1=model1.predict(X_test)

model2=svm.SVC(kernel="poly",degree=3)
model2.fit(X_train,y_train)
pred2=model2.predict(X_test)

model3=svm.SVC(kernel="rbf")
model3.fit(X_train,y_train)
pred3=model3.predict(X_test)

for name,pred in [("Linear",pred1),("Poly",pred2),("RBF",pred3)]:
  print("Model:",name)
  print("Accuracy Score:",accuracy_score(y_test,pred))