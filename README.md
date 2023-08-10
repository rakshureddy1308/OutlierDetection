# OutlierDetection

### Load the dataset 
from sklearn.datasets import load_diabetes
a = load_diabetes()
X = a.data

### Import the required libraries

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)

y_pred = clf.predict(X)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest Outlier Detection on DIABETES Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

![4](https://github.com/rakshureddy1308/OutlierDetection/assets/119916578/28837356-2984-4f9f-a5bc-ffeeac8e6633)


from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Local Outlier Factor Outlier Detection on DIABETES Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

![5](https://github.com/rakshureddy1308/OutlierDetection/assets/119916578/598d3e99-c554-4eab-b1e6-b07ccea8f6c2)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

### Load the IRIS dataset
a = load_diabetes()
X = a.data
y = a.target

### Fit Isolation Forest model
clf_iso = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = clf_iso.fit_predict(X)

### Fit Local Outlier Factor model
clf_lof = LocalOutlierFactor(contamination=0.1)
y_pred_lof = clf_lof.fit_predict(X)

### Plot Isolation Forest outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_iso == -1, 'red', 'blue'), label='Isolation Forest')
plt.title("Outlier Detection using Isolation Forest on DIABETES Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

![6](https://github.com/rakshureddy1308/OutlierDetection/assets/119916578/41855cfc-b5cf-4245-a7aa-da2e36a10be5)


### Plot Local Outlier Factor outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_lof == -1, 'red', 'blue'), label='Local Outlier Factor')
plt.title("Outlier Detection using Local Outlier Factor on DIABETES Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

![7](https://github.com/rakshureddy1308/OutlierDetection/assets/119916578/48f9ef12-713f-4fd7-92d4-9e171f9dc34b)

