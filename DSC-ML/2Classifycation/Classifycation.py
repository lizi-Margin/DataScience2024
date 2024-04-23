import numpy as np
 
class KNN :
    def __init__(self,n_neighbors = 5) -> None:
        self.X = None
        self.y = None
        self.fited = False
        self.k = n_neighbors
        
    def fit(self,X,y):
        self.X = X
        self.fited = True
        self.y = y
    
    def predict_one (self,x):
        # 大写参数表示常量，不可改变
        distances = np.sqrt(np.sum(np.square(self.X - x), axis=1))
        # 计算距离矩阵
        len_dis = len(distances)
        # 得到distances数目
        labels = []
        # 存储标签
        for i in range(0, self.k):
            min_value = distances[i]
            min_value_idx = i
            for j in range(i + 1, len_dis):
                if distances[j] < min_value:
                    min_value = distances[j]
                    min_value_idx = j
            distances[i], distances[min_value_idx] = distances[min_value_idx], distances[i]
            labels.append(self.y[min_value_idx])
        # 选择排序挑选出前k个最值
        # 用labels存储前k个最小距离的标签
        C = labels[0]
        max_count = 0
        for label in labels:
            count = labels.count(label)
            if count > max_count:
                max_count = count
                C = label
        # 求前k个label中，重复次数最多的label，并返回
        return C
    def predict(self,X_t):
        if not self.fited :
            print('err') 
            return None
        res = []
        for x in X_t:
           res.append (self.predict_one(x)) 
            
        return np.array(res)
        
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


class GNB: # 高斯分布处理连续特征，目前只能分两类
    def __init__(self) -> None:
        self.mean = 0
        self.std = 0

        self.p0_prior = 0
        self.p1_prior = 0
        self.mean_0 = 0
        self.mean_1 = 0
        self.std_0 =0
        self.std_1 =0
        self.fitted = False



    def  fit(self,X,y):
        self.fitted = True



        # 计算先验概率
        training_mat, training_labels = X,y
        feature_size = len(training_mat[0])
        training_size = len(training_mat)
        self.p1_prior = (sum(training_labels) + 1) / (float(training_size) + 2)
        self.p0_prior = 1 - self.p1_prior
        # 计算条件概率
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)
        i1,i0 = 0,0 
        features_1 = [0]*training_size
        features_0 = [0]*training_size
        for i in range(training_size):
            if training_labels[i] == 1:
                features_1[i1] =  training_mat[i]
                i1+=1
            else:
                features_0[i0] =  training_mat[i]
                i0+=1
        features_0 = np.array(features_0[:i0])
        features_1 = np.array(features_1[:i1])

        self.mean_1 = np.mean(features_1,axis=0)
        self.mean_0 = np.mean(features_0,axis=0)
        self.std_0 = np.std(features_0,axis=0)
        self.std_1 = np.std(features_1,axis=0)

    def predict(self,X_t):
        if not self.fitted :
            print('err') 
            return None
        res = []
        for x in X_t:
            
           res.append (self.predict_one(x)) 
            
        return np.array(res)
 

    def predict_one (self,test_data):
        feature_size = len(test_data)
        # 计算目标函数
        p1_pred, p0_pred = self.p1_prior, self.p0_prior    
        for i in range(feature_size):
                p1_pred *= GNB.normal_dist(test_data[i],self.mean_1[i],self.std_1[i])                
                p0_pred *= GNB.normal_dist(test_data[i],self.mean_0[i],self.std_0[i])
        if p1_pred > p0_pred:
            return 1
        else:
            return 0

    @staticmethod 
    def normal_dist(x , mean , sd):
        prob_density = (1/sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density

