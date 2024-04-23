from Classifycation import * 
import sklearn.datasets  
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as sk_KNN

# 导入数据集
bc = sklearn.datasets.load_breast_cancer()
X = bc.data
y =  bc.target


# 创建Logistic回归模型
knn = KNN(n_neighbors=7) # k = 7
sk_knn = sk_KNN() # sklearn knn 默认k值为5

# 划分数据集100次 ,求平均准确率
n = 100
avg_accuracy = 0
sk_avg_accuracy = 0
for i in range(n):
    X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.3)
    # 训练模型
    knn.fit(X_train, y_train)
    sk_knn.fit(X_train, y_train)

    # 预测测试集
    y_pred = knn.predict(X_test)
    sk_y_pred = sk_knn.predict(X_test)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    sk_accuracy = accuracy_score(y_test, sk_y_pred)
    avg_accuracy += accuracy
    sk_avg_accuracy +=sk_accuracy
avg_accuracy /= n
sk_avg_accuracy /= n

print("Accuracy:", avg_accuracy)
print("sk_Accuracy:", sk_avg_accuracy)
