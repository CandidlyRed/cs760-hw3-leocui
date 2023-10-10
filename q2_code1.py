import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm

data=pd.read_csv('./hw3Data/D2z.txt',sep=' ',header=None)
print(data)
x=data.iloc[:,0:2].values
y=data.iloc[:,2].values

x1,x2=np.arange(-2,2.1,0.1),np.arange(-2,2.1,0.1)

xx, yy = np.meshgrid(x1, x2)
xtest = np.vstack((np.ravel(xx), np.ravel(yy))).T

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)

ypred=knn.predict(xtest)

plt.scatter(xtest[:,0],xtest[:,1],marker="x",c=ypred, cmap=cm.get_cmap('Wistia', len(set(y))))
plt.scatter(x[:,0],x[:,1],c=y,marker="o", cmap=cm.get_cmap('coolwarm', len(set(ypred))))
plt.show()