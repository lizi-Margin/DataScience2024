from Cluster import*
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as sk_KMeans
import sklearn.datasets  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 导入数据集
bc = sklearn.datasets.load_breast_cancer()
X = bc.data
y =  bc.target


# 创建模型
kmeans = KMeans(2) # k = 7
sk_kmeans = sk_KMeans(2) # sklearn knn 默认k值为5

# 划分数据集100次 ,求平均准确率
n = 10
avg_accuracy = 0
sk_avg_accuracy = 0
y_pred = None
X_test = None
y_test = None
for i in range(n):
    X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.2)
    # 训练模型
    kmeans.fit(X_train)
    sk_kmeans.fit(X_train)

    # 预测测试集
    y_pred = KMeans.set_lable_0_1(kmeans.predict(X_test),y_test)
    sk_y_pred = KMeans.set_lable_0_1(sk_kmeans.predict(X_test),y_test)
    # y_pred = (kmeans.predict(X_test))
    # sk_y_pred =(sk_kmeans.predict(X_test))

    # print(y_pred,sk_y_pred,y_test)


    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    sk_accuracy = accuracy_score(y_test, sk_y_pred)
    avg_accuracy += accuracy
    sk_avg_accuracy +=sk_accuracy
avg_accuracy /= n
sk_avg_accuracy /= n

print("Accuracy:", avg_accuracy)
print("sk_Accuracy:", sk_avg_accuracy)

plt.scatter(kmeans.clusterCents[:,0].tolist(),kmeans.clusterCents[:,1].tolist(),c='r',marker='^')
plt.scatter(X_test[:,0],X_test[:,1],c=y_pred*100+20,linewidths=np.power(y_pred+0.5, 2),alpha=0.5)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test*100+20,linewidths=np.power(y_pred+0.5, 2),alpha=0.5)
plt.show()
