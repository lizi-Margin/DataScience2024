from Regression import*
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.datasets import fetch_openml 
import matplotlib.pyplot as plt


# 一元线性回归
# 导入数据集
boston = fetch_openml(name='boston', version=1)
X = boston.data.RM.values.astype(np.float64)
X2 = boston.data.values .astype(np.float64)# 使用CRIM-LSTAT作为X
y = boston.target.values 

# 进行回归计算
# 自制回归方法
lr=LRegression(6.3e-6,int(150e3))
lr.fit(X2,y)
price_p = lr.predict(X2)
mse = mean_squared_error(y,price_p)
# sklearn线性回归方法
sk_lr = LinearRegression()
sk_lr.fit(X2,y)
sk_price_p = sk_lr.predict(X2)
sk_mse = mean_squared_error(y,sk_price_p)

# 打印结果
print ('Lr_mse = ',mse,'sk_Lr_mse = ',sk_mse,'差距：',mse  -  sk_mse )
x = X
fig, ax = plt.subplots()
plt.xlim(x.min(),1.1*x.max())
ax.scatter(x,y,alpha=0.2)
ax.scatter(x,price_p,color = 'red',alpha = 0.8)
ax.scatter(x,sk_price_p,color = 'blue',alpha = 0.8)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('LRegression(red) of Mine & SklearnLinRegression(blue) Comparison Chart')
plt.show()