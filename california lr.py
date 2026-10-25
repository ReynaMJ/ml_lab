import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load: one feature (AveRooms) -> price
data = fetch_california_housing()
x = data.data[:, 2]            # column 2 = average rooms
y = data.target
keep = x < 10                  # remove crazy outliers
x, y = x[keep], y[keep]

# 2. Preprocess: scale x, add a column of 1s for the intercept w0
x_scaled = (x - x.mean()) / x.std()
X = np.c_[np.ones(len(x)), x_scaled]

# 3a. Normal equation: w = (XᵀX)⁻¹ Xᵀ y
w_normal = np.linalg.inv(X.T @ X) @ X.T @ y

# 3b. Gradient descent: repeat w = w - lr * gradient
w = np.zeros(2)
lr = 0.1
for i in range(1000):
    error = X @ w - y
    gradient = X.T @ error / len(y)
    w = w - lr * gradient

# 4. Evaluate
pred = X @ w
print("Normal eq weights:", w_normal)
print("Grad desc weights:", w)          # should be almost the same
print("MSE:", mean_squared_error(y, pred))
print("R2 :", r2_score(y, pred))

# 5. Plot
plt.scatter(x, y, s=1)
plt.plot(x, pred, color='red')
plt.xlabel("Average rooms"); plt.ylabel("Price"); plt.show()