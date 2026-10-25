#Question 7
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report

data=fetch_20newsgroups(subset="all",remove=("headers","footers","quotes"))

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#multinomial conversion of features
vectorizer_multi=CountVectorizer(stop_words="english",max_features=5000)
X_train_m=vectorizer_multi.fit_transform(X_train)
X_test_m=vectorizer_multi.transform(X_test)

#bernoulli conversion of features
vectorizer_bern=CountVectorizer(stop_words="english",binary=True,max_features=5000)
X_train_b=vectorizer_bern.fit_transform(X_train)
X_test_b=vectorizer_bern.transform(X_test)

#training of multinomial  model
model1=MultinomialNB()
model1.fit(X_train_m,y_train)
y_pred1=model1.predict(X_test_m)

#training of bernoulli model
model2=BernoulliNB()
model2.fit(X_train_b,y_train)
y_pred2=model2.predict(X_test_b)

for name,pred in [("Multinomial",y_pred1),("Bernoulli",y_pred2)]:
  print("Accuracy score:",accuracy_score(y_test,pred))
  print("F1 Score:",f1_score(y_test,pred,average='macro'))
  print("Classification report:\n",classification_report(y_test,pred))