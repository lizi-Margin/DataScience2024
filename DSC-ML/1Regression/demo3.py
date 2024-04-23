from Regression import LogisticRegression
import sklearn.datasets  
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.metrics import accuracy_score

# 导入数据集
bc = sklearn.datasets.load_breast_cancer()
X = bc.data
y =  bc.target

# 创建Logistic回归模型
logreg = LogisticRegression(lr= 5e-2,n=int(4e3))
sk_logreg = sk_LogisticRegression()

# 训练模型
logreg.fit(X, y)
sk_logreg.fit(X, y)

# 预测测试集
y_pred = logreg.predict(X)
sk_y_pred = sk_logreg.predict(X)

# 计算预测准确率
accuracy = accuracy_score(y, y_pred)
sk_accuracy = accuracy_score(y, sk_y_pred)
print("Accuracy:", accuracy)
print("sk_Accuracy:", sk_accuracy)
