import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./CW_Data.csv', sep=',', header=0)
data = df.values
X = data[:, 1:6]
Y = data[:, 6]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
train_score = knn.score(X_train, Y_train)
test_score = knn.score(X_test, Y_test)
print("Accuracy", test_score)
X1 = np.array([[30,1,10,5,0], [14,7,10,2,0]])
prediction = knn.predict(X1)
k = data[:, 6][prediction]
print("The program of the first：", k[0])
print("The program of the second：", k[1])

