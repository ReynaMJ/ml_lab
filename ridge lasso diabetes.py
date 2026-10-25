from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)            # already scaled
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

models = {
    "Linear": LinearRegression(),
    "Ridge":  RidgeCV(alphas=[0.01, 0.1, 1, 10]),   # tries each alpha with CV
    "Lasso":  LassoCV(cv=5),                         # 5-fold CV picks alpha
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MSE:", mean_squared_error(y_test, pred), "R2:", r2_score(y_test, pred))

print("Best Ridge alpha:", models["Ridge"].alpha_)
print("Best Lasso alpha:", models["Lasso"].alpha_)
print("Lasso zero weights:", (models["Lasso"].coef_ == 0).sum())