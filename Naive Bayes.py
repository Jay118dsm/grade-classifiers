import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#因为所有特征之间相互独立，所以可以用朴素贝叶斯算法

#读取数据
df = pd.read_csv("./CW_Data.csv" , sep=',', header=0)
data = df.values
X = data[:, 1:6]
Y = data[:, 6]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)#测试集占0.2，随机数种子为0

#朴素贝叶斯算法预估器（naive bayes）
estimator = MultinomialNB(alpha=1)
estimator.fit(X_train, Y_train)

#模型评估
#比对真实值和预测值
y_predict = estimator.predict(X_test)
print("y_predict:\n", y_predict)
print("对比\n",  Y_test == y_predict)
#计算准确率
train_score = estimator.score(X_train, Y_train)
test_score = estimator.score(X_test, Y_test)
print("Accuracy", test_score)

