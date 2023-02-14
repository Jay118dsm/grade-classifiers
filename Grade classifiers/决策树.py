import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from IPython.display import Image


#读取数据
df = pd.read_csv("./CW_Data.csv" , sep=',', header=0)
data = df.values
X = data[:, 1:6]
Y = data[:, 6]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)#测试集占0.2，随机数种子为0
#可以不用做标准化
#决策树算法预估器（decision tree）
estimator = DecisionTreeClassifier(criterion='entropy' , max_depth= 5)
#选择信息增益的熵作为参考，树深度为5可以防止过拟合（数据较少可以不用管）,为了方便画图能看清内容，取了3
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

#可视化，
#导入到 http://webgraphviz.com/ 转换为树状图
#pic = export_graphviz(estimator, out_file="decision_tree.dot", feature_names= ['Q1','Q2','Q3','Q4','Q5'])
#或
plt.figure(figsize=(15,9))
plot_tree(estimator,filled=True,feature_names=['Q1','Q2','Q3','Q4','Q5'])
plt.show()
#或
# tmp_dot_file = 'decision_tree_tmp.dot'
# export_graphviz(estimator, out_file=tmp_dot_file, feature_names=['Q1','Q2','Q3','Q4','Q5'],filled=True, impurity=False)
# with open(tmp_dot_file) as f:
#     dot_graph = f.read()
# graph = pydotplus.graph_from_dot_data(dot_graph)
# graph.write_pdf('example.pdf')    #保存图像为pdf格式
# Image(graph.create_png())   #绘制图像为png格式
