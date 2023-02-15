import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('./CW_Data.csv', sep=',', header=0)
data = df.values
X = data[:, 1:6]
Y = data[:, 6]
Class_0 = []
Class_1 = []
Class_2 = []
Class_3 = []
Class_4 = []
for i in range(len(data)):
    if data[i,-1] == 0:
        Class_0.append(data[i,1:-1])
    if data[i,-1] == 1:
        Class_1.append(data[i,1:-1])
    if data[i,-1] == 2:
        Class_2.append(data[i,1:-1])
    if data[i,-1] == 3:
        Class_3.append(data[i,1:-1])
    if data[i,-1] == 4:
        Class_4.append(data[i, 1:-1])
plt.figure()
plt.title('k-means')
pca = PCA(n_components=2)
dataset = np.concatenate([Class_0, Class_1, Class_2, Class_3, Class_4], axis=0)
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
afterpca = pca.fit_transform(dataset)
afterpca_train, afterpca_test = train_test_split(afterpca, test_size=0.30, random_state=0)
kmeans = KMeans(7, random_state=0)
kmeans.fit(afterpca)
labels = kmeans.predict(afterpca)

clusters = [2,3,4,5,6,7]
s1 = [0.41738186915552794, 0.467969846421172, 0.4203233544227555, 0.43402849018097167, 0.4152332613090985, 0.4056698574253383]

print(kmeans.cluster_centers_)
print(silhouette_score(afterpca,kmeans.labels_))
plt.scatter(afterpca[:, 0], afterpca[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

plt.figure()
plt.plot(clusters, s1, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()


