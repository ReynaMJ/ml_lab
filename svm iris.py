import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

df=datasets.load_iris()

X=df.data[:,:2]
y=df.target

y=np.where(y==0,1,-1)

model=svm.SVC(kernel="linear",C=1)
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y)

#get the current axes
ax=plt.gca()
#getting the limit of current axes
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],100)
yy=np.linspace(ylim[0],ylim[1],100)

YY,XX=np.meshgrid(yy,xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z=model.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,Z,levels=[0])

ax.contour(XX,YY,Z,levels=[-1,1],linestyles=['--','--'])

ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=100,facecolor='none',edgecolor='k')

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris dataset")
plt.show()

