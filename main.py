import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./CW_Data.csv', sep=',', header=0)
print(type(df))
print(type(df.values))
# print(df.values.shape)
data = df.values
#
# # 学生选课分布
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
plt.title('Program Count')
plt.bar(['0','1','2','3','4'],[len(Class_0),len(Class_1),len(Class_2),len(Class_3),len(Class_4)])
plt.show()
# # #
# # # 均值
plt.figure()
plt.title('Average')
plt.scatter(np.array(Class_0).mean(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).mean(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).mean(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).mean(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).mean(-1),np.zeros(len(Class_4))+4)
plt.show()
# #
# # # 标准差
plt.figure()
plt.title('Standard Deviation')
plt.scatter(np.array(Class_0).std(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).std(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).std(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).std(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).std(-1),np.zeros(len(Class_4))+4)
plt.show()
# #
# # # 最大值
plt.figure()
plt.title('Max')
plt.scatter(np.array(Class_0).max(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).max(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).max(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).max(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).max(-1),np.zeros(len(Class_4))+4)
plt.show()
# #
# # # 最小值
plt.figure()
plt.title('Min')
plt.scatter(np.array(Class_0).min(-1),np.zeros(len(Class_0)))
plt.scatter(np.array(Class_1).min(-1),np.zeros(len(Class_1))+1)
plt.scatter(np.array(Class_2).min(-1),np.zeros(len(Class_2))+2)
plt.scatter(np.array(Class_3).min(-1),np.zeros(len(Class_3))+3)
plt.scatter(np.array(Class_4).min(-1),np.zeros(len(Class_4))+4)
plt.show()
# #
# # # t-SNE
# TSNE
plt.figure()
plt.title('t-SNE')
tsne = TSNE(n_components=2)
dataset = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
tsne.fit_transform(dataset)
newdataset = tsne.embedding_
for i in range(len(Class_0)):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='blue')
for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='c')
for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='g')
for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='k')
for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
    plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, c='m')
plt.show()
# #
# PCA
plt.figure()
plt.title('PCA')
pca = PCA(n_components=2)
dataset = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
afterpca = pca.fit_transform(dataset)
for i in range(len(Class_0)):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='blue')
for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='c')
for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='g')
for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='k')
for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
    plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, c='m')
plt.show()
# #
# LDA
plt.figure()
plt.title('LDA')
lda = LDA(n_components=2)
train_x = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
Class_0_y = np.zeros(len(Class_0))
Class_1_y = np.zeros(len(Class_1))+1
Class_2_y = np.zeros(len(Class_2))+2
Class_3_y = np.zeros(len(Class_3))+3
Class_4_y = np.zeros(len(Class_4))+4
train_y = np.concatenate([Class_0_y,Class_1_y,Class_2_y,Class_3_y,Class_4_y], axis=0)
afterlda = lda.fit_transform(train_x,train_y)
for i in range(len(Class_0)):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='blue')
for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='c')
for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='g')
for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5, c='k')
for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
    plt.scatter(newdataset[i][0], newdataset[i][1], alpha=0.5, c='m')
plt.show()


