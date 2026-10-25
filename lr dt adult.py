#Quesion 10

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns=['age','workclass','fnlwgt','education','education_no','marital_status','occupation',
         'relationship','race','sex','capital_gain','capital_loss','native_country','income']

df=pd.read_csv(url,names=columns,skipinitialspace=True,na_values='?')

#preprocessing
df.dropna(inplace=True)

le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
  df[col]=le.fit_transform(df[col])

X=df.drop('income',axis=1)
y=df['income']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#training lr model
model1=LogisticRegression(max_iter=1000)
model1.fit(X_train_scaled,y_train)
y_pred1=model1.predict(X_test_scaled)

#training DT model
model2=DecisionTreeClassifier(criterion='entropy',max_depth=5)
model2.fit(X_train_scaled,y_train)
y_pred2=model2.predict(X_test_scaled)

for name,pred in [('Logistic',y_pred1),("Decision tree",y_pred2)]:
  print("Model:",name)
  print("Accuracy Score:",accuracy_score(y_test,pred))
  print("Classification report:\n",classification_report(y_test,pred))
  print()

