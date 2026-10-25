import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = sns.load_dataset('mpg').dropna()          # remove missing values
x = data[['displacement']].values
y = data['mpg'].values

x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)   # for smooth curve

for degree in [1, 2, 3, 5]:                      # 1 = linear regression
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    pred = model.predict(x)
    print("degree", degree, "MSE", mean_squared_error(y, pred), "R2", r2_score(y, pred))
    plt.plot(x_line, model.predict(x_line), label=f"degree {degree}")

plt.scatter(x, y, s=5, color='gray')
plt.xlabel("Displacement"); plt.ylabel("MPG"); plt.legend(); plt.show()