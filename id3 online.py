#Question 9

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

df = pd.read_excel(url,nrows=50000)

df=df.dropna(subset=["CustomerID"])

df=df[(df["Quantity"]>0) & (df["UnitPrice"]>0)]

df["TotalSpend"]=df["Quantity"]*df["UnitPrice"]

customer_df=df.groupby("CustomerID").agg({
    'Quantity':'sum',
    'UnitPrice':'median',
    'TotalSpend':'sum',
    'Country':'first'
}).reset_index()

thresh=customer_df['TotalSpend'].median()
customer_df['HighValue']=(customer_df['TotalSpend']>thresh).astype('int')

le=LabelEncoder()
customer_df['Country'] = le.fit_transform(customer_df['Country'])

X=customer_df[['Country','Quantity','UnitPrice']]
y=customer_df['HighValue']

model=DecisionTreeClassifier(criterion="entropy",max_depth=4)
model.fit(X,y)

plt.figure(figsize=(20,8))

plot_tree(model,feature_names=X.columns,class_names=['Low','High'],filled=True)
plt.title("Customer Segmentation using ID3")
plt.show()

feature_imp=pd.Series(model.feature_importances_,index=X.columns)
print("Feature Importance:\n",feature_imp.sort_values(ascending=False))

