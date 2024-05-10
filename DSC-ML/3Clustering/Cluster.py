import numpy as np
import copy

class KMeans:
    def __init__(self, k_num) -> None:
        self.k = k_num
        self.clusterCents = None

    @ staticmethod
    def L2(vecXi, vecXj):
        '''
        计算欧氏距离
        para vecXi：点坐标，向量
        para vecXj：点坐标，向量
        retrurn: 两点之间的欧氏距离
        '''
        return np.sqrt(np.sum(np.power(vecXi - vecXj, 2)))

    def fit(self,S):
        '''
        K均值聚类
        para S：样本集，多维数组
        para k：簇个数
        para distMeas：距离度量函数，默认为欧氏距离计算函数
        return sampleTag：一维数组，存储样本对应的簇标记
        return clusterCents：一维数组，各簇中心
        retrun SSE:误差平方和
        '''
        distMeas= KMeans.L2
        # print('k = ' , self.k)
        m = np.shape(S)[0] # 样本总数
        sampleTag = np.zeros(m).astype(int)
        # print('sampleTag.shape=',sampleTag)
        # 随机产生k个初始簇中心
        n = np.shape(S)[1] # 样本向量的特征数
        # print('n = ' , n)
        # self.clusterCents = np.mat([[-1.93964824,2.33260803],[7.79822795,6.72621783],[10.64183154,0.20088133]])
        self.clusterCents = np.mat(np.zeros((self.k,n)))
        for j in range(n):
            minJ = min(S[:,j]) 
            rangeJ = float(max(S[:,j]) - minJ)
            self.clusterCents[:,j] = np.mat(minJ + rangeJ * np.random.rand(self.k,1))
            
        sampleTagChanged = True
        SSE = 0.0
        while sampleTagChanged: # 如果没有点发生分配结果改变，则结束
            sampleTagChanged = False
            SSE = 0.0
            
            # 计算每个样本点到各簇中心的距离
            # m是样本总数
            for i in range(m):
                minD = np.inf
                minIndex = -1
                # k是簇中心个数
                for j in range(self.k):
                    # S样本集，clusterCents样本中心点
                    d = distMeas(self.clusterCents[j,:],S[i,:])
                    if d < minD:
                        minD = d
                        minIndex = j
                if sampleTag[i] != minIndex: 
                    sampleTagChanged = True
                sampleTag[i] = minIndex
                SSE += minD**2
           
            # 重新计算簇中心
            for i in range(self.k):
                ClustI = S[np.nonzero(sampleTag[:]==i)[0]]
                self.clusterCents[i,:] = np.mean(ClustI, axis=0)

        return self.clusterCents, sampleTag, SSE
    
    def  predict(self,S):
        distMeas =  KMeans.L2
        SSE = 0.0
        m = np.shape(S)[0] # 样本总数
        sampleTag = np.zeros(m).astype(int)       
        # 计算每个样本点到各簇中心的距离
        # m是样本总数
        for i in range(m):
            minD = np.inf
            minIndex = -1
            # k是簇中心个数
            for j in range(self.k):
                # S样本集，clusterCents样本中心点
                d = distMeas(self.clusterCents[j,:],S[i,:])
                if d < minD:
                    minD = d
                    minIndex = j

            sampleTag[i] = minIndex
            SSE += minD**2
        
        return sampleTag
    
    @staticmethod
    def set_lable_0_1 (sampleTag , y):
        n = min(len(sampleTag),len(y))
        correct_sum = 0
        for i in range(n):
            if (sampleTag[i]==y[i]):
                correct_sum +=1
        if (correct_sum < n/2):
            sampleTag = KMeans.invert_0_1(sampleTag)
        return sampleTag
    
    @staticmethod
    def set_lable_0_1_2 (sampleTag , y):
        n = min(len(sampleTag),len(y))
        
        # class_num = KMeans.get_class_num(sampleTag+y)
        class_num = np.array([0,1,2]).astype(int)

        

        highest_correct_sum = -1
        ret = None
        for ii in class_num:
            class_num_n = copy.deepcopy(class_num)
            class_num_n = KMeans.arr_drop_num(class_num_n,ii)
            for jj in class_num_n:

                
                new_sampleTag = copy.deepcopy(sampleTag)
                new_sampleTag = KMeans.set_num(new_sampleTag,0,-1)
                new_sampleTag = KMeans.set_num(new_sampleTag,1,-2)
                new_sampleTag = KMeans.set_num(new_sampleTag,2,-3)

                new_sampleTag = KMeans.set_num(new_sampleTag,-1,ii)
                new_sampleTag = KMeans.set_num(new_sampleTag,-2,jj)
                class_num_new = copy.deepcopy(class_num)
                class_num_new = KMeans.arr_drop_num(class_num_new,ii)
                class_num_new = KMeans.arr_drop_num(class_num_new,jj)                
                new_sampleTag = KMeans.set_num(new_sampleTag,-3,class_num_new[0])
                print(new_sampleTag,ii,jj,class_num_new[0])

                correct_sum = 0
                for i in range(n):
                    if (new_sampleTag[i]==y[i]):
                        correct_sum +=1
                if (correct_sum > highest_correct_sum):
                    highest_correct_sum = correct_sum
                    ret = new_sampleTag
 
        return ret
    
    @staticmethod 
    def get_class_num(tag):
        class_num=[]
        for num in tag :
            if not KMeans.arr_has_num(class_num,num):
                class_num.append(num)
        return np.array(class_num)

    @staticmethod 
    def arr_has_num(arr, num):
        for n in arr:
            if n == num:
                return True
        return False

    @staticmethod 
    def arr_drop_num(arr, num):
        arr_1 = []
        for i in range (len(arr)):
            if (arr[i] != num ):
                arr_1.append(arr[i])            
       
        return np.array(arr_1)



    @staticmethod 
    def set_num(arr, from_ :int, to_ : int ):
        for i in range (len(arr)):
            if (arr[i] == from_ ):
                arr[i] = to_             
       
        return arr

    def switch_num(arr, a:int, b : int ):
        for i in range (len(arr)):
            if (arr[i] == a ):
                arr[i] =  b            
            elif (arr[i] == b ):
                arr[i] =  a            
       
        return arr

     

    @staticmethod
    def invert_0_1( arr ):
        for i in range (len(arr)):
            if (arr[i] == 0 ):
                arr[i] = 1 
            
            else :
                arr[i] =1
        
        return arr


