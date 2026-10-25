import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load and preprocess (labels are NOT used for clustering)
X, y = load_digits(return_X_y=True)          # 1797 images, 64 pixels each
X = StandardScaler().fit_transform(X)

# 2. Try different K
ks = range(2, 16)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, model.labels_))
    print(f"K={k:2d} | inertia={model.inertia_:.0f} | silhouette={silhouettes[-1]:.3f}")

# 3. Plot both metrics against K
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(ks, inertias, 'o-');    ax1.set_title("Inertia (look for the elbow)")
ax2.plot(ks, silhouettes, 'o-'); ax2.set_title("Silhouette (look for the peak)")
for ax in (ax1, ax2): ax.set_xlabel("K")
plt.show()