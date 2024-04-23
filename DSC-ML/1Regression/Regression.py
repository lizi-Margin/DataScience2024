import numpy as np
class LRegression:
    #初始化参数，k为斜率，b为截距，a为学习率，n为迭代次数
    def __init__(self,a,n):
        self.k = None
        self.b=None
        self.a=a
        self.n=n
    
    # 多元线性回归
    #梯度下降法迭代训练模型参数
    def fit(self,X,y):
        X = np.array(X)
        if (X.ndim != 2) : 
            print("ndim error!")
            return
        X_shape = np.shape(X)[1]       

        self.k = np.ones(X_shape)
        self.b = 0
        

        #计算总数据量
        m=len(X)
        #循环n次
        b_grad = 1
        k_grad = np.zeros(X_shape)
        for _ in range ( self.n):
            b_grad=0
            k_grad=np.zeros(X_shape)
            #计算梯度的总和再求平均
            for j in range(m):
                kXj_sum = 0
                for kk in range(X_shape):
                    
                    kXj_sum += ((self.k[kk]*(X[j][kk])))
                for ii in range(X_shape):
                    k_grad[ii] += (1/m)*((kXj_sum+self.b)-y[j])*X[j][ii]
                b_grad += (1/m)*((kXj_sum+self.b)-y[j])

            #更新k,b
            self.b=(1-self.a)*self.b-(self.a*b_grad)
            self.k=(1-self.a)*self.k-(self.a*k_grad)

        self.params= {'k':self.k,'b':self.b}

        return self.params    
    #预测函数
    def predict(self,X):
        # 输入合法性检测
        X = np.array(X)
        if (X.ndim != 2) : 
            print("ndim error!")
            return 
        X_shape = np.shape(X)[1] 
        # res
        res = np.zeros(len(X))
        # 通过参数k,b计算预测结果
        for ii in range(len(X)):
            kXj_sum = 0
            for kk in range(X_shape):
                kXj_sum += ((self.k[kk]*X[ii][kk]))
            y_pred =kXj_sum + self.b
            res[ii] = y_pred
        return res


    # 一元线性回归
    #梯度下降法迭代训练模型参数
    def fit_one(self,x,y,k=1,b=0):
        self.k = k
        self.b = b
        #计算总数据量
        m=len(x)
        #循环n次
        for i in range(self.n):
            b_grad=0
            k_grad=0
            #计算梯度的总和再求平均
            for j in range(m):
                b_grad += (1/m)*((self.k*x[j]+self.b)-y[j])
                k_grad += (1/m)*((self.k*x[j]+self.b)-y[j])*x[j]

            #更新k,b
            self.b=self.b-(self.a*b_grad)
            self.k=self.k-(self.a*k_grad)
        # 输出参数
        self.params= {'k':self.k,'b':self.b}
        return self.params
    
    #预测函数
    def predict_one(self,x):
        y_pred =self.k * x + self.b
        return y_pred



class LogisticRegression:
    def __init__(self, lr=0.01, n=1000):
        self.learning_rate = lr
        self.num_iterations =n 
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 初始化权重和偏置
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降
        for _ in range(self.num_iterations):
            linear_model = (np.dot(X, self.weights)) + self.bias
            y_pred = LogisticRegression.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = LogisticRegression.sigmoid(linear_model)
        return y_pred

    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_prob(X)
        y_pred = np.zeros_like(y_pred_prob)
        y_pred[y_pred_prob >= threshold] = 1
        return y_pred

    @staticmethod
    def sigmoid(x):
        if len(x) <=1:
            if x>0:
                return 1.0/(1.0+np.exp(-x))
            else:
                return np.exp(x)/(1.0+np.exp(x))
        else :
            for i in range (len (x)):
             if x[i]>0:
                x[i] = 1.0/(1.0+np.exp(-x[i]))
             else:
                x[i] =  np.exp(x[i])/(1.0+np.exp(x[i]))   

            return x
