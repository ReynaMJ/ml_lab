import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
data = pd.read_csv(url, header=None)
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Without scaling
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print("WITHOUT scaling\n", classification_report(y_test, model.predict(X_test)))

# With scaling (fit scaler on train only!)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)
print("WITH scaling\n", classification_report(y_test, model.predict(scaler.transform(X_test))))