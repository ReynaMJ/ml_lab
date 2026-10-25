from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

models = {
    "MLE (no penalty)": LogisticRegression(penalty=None, max_iter=5000),
    "MAP L2 (Gaussian)": LogisticRegression(penalty='l2', C=0.1),
    "MAP L1 (Laplace)":  LogisticRegression(penalty='l1', C=0.1, solver='liblinear'),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name,
          "| accuracy:", round(model.score(X_test, y_test), 3),
          "| sum of |weights|:", round(abs(model.coef_).sum(), 2),
          "| zero weights:", (model.coef_ == 0).sum())