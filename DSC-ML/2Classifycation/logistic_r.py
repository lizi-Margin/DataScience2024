from Classifycation import * 
import sklearn.datasets  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

# 导入数据集
bc = sklearn.datasets.load_breast_cancer()
X = bc.data
y =  bc.target

# 创建Logistic回归模型
logreg = LogisticRegression(lr= 5e-2,n=int(4e3))
sk_logreg = sk_LogisticRegression()

# 划分数据集100次 ,求平均准确率
n = 100
avg_accuracy = 0
sk_avg_accuracy = 0
for i in range(n):
    X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size=0.3)
    # 训练模型
    logreg.fit(X_train, y_train)
    sk_logreg.fit(X_train, y_train)

    # 预测测试集
    y_pred = logreg.predict(X_test)
    sk_y_pred = sk_logreg.predict(X_test)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    sk_accuracy = accuracy_score(y_test, sk_y_pred)
    avg_accuracy += accuracy
    sk_avg_accuracy +=sk_accuracy
avg_accuracy /= n
sk_avg_accuracy /= n

print("Accuracy:", avg_accuracy)
print("sk_Accuracy:", sk_avg_accuracy)
