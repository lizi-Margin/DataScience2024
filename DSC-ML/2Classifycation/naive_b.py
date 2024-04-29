from Classifycation import * 
import sklearn.datasets  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn. naive_bayes import GaussianNB as sk_NB

# 导入数据集
bc = sklearn.datasets.load_breast_cancer()
X = bc.data
y =  bc.target


# 创建Naive Bayers回归模型
nb = GNB()
sk_nb = sk_NB() # sklearn Naive Bayes 

# 划分数据集100次 ,求平均准确率
n = 100
avg_accuracy = 0
sk_avg_accuracy = 0
for i in range(n):
    X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.3)
    # 训练模型
    nb.fit(X_train, y_train)
    sk_nb.fit(X_train, y_train)

    # 预测测试集
    y_pred = nb.predict(X_test)
    sk_y_pred = sk_nb.predict(X_test)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    sk_accuracy = accuracy_score(y_test, sk_y_pred)
    avg_accuracy += accuracy
    sk_avg_accuracy +=sk_accuracy
avg_accuracy /= n
sk_avg_accuracy /= n

print("Accuracy:", avg_accuracy)
print("sk_Accuracy:", sk_avg_accuracy)
