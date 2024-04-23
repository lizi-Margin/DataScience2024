import sklearn .impute as impute

# 读取数据集
import pandas as pd
data = (pd.read_csv('house_train.csv')).drop('id',axis=1)
print("房产平均价： ",data['price'].mean())
print("房产平均占地面积： ",data['area'].mean())




######################################################################################################
# 缺失值的检测与处理


# 查看缺失率
total =  data.isnull().sum().sort_values(ascending=False)
null_percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data =  pd.concat([total,null_percent],axis=1,keys=['Total','Percent'])
print(missing_data)

# 删除-字符串null
# imputer =  impute.SimpleImputer(missing_values='',strategy="drop")
# imputer.fit(data)
# imputer.transform(data)
# 填充平均值
means = data.mean()
data = data.fillna(means)


#################################################################################
# 特征编码

import  sklearn.preprocessing as pre
feature_encoder_floor = pre.LabelEncoder()
floor =data['floor']
feature_encoder_floor.fit(floor)
data['floor'] = feature_encoder_floor.transform(floor)

feature_encoder_built_date = pre.LabelEncoder()
built_date =data['built_date']
feature_encoder_built_date.fit(built_date)
# print(feature_encoder_built_date.classes_)
data['built_date'] = feature_encoder_built_date.transform(built_date)






###########################################################################################
# 异常值检测
import matplotlib.pyplot as plt 
fig = plt.figure(figsize = (10,6)) 
fig.add_subplot(2,1,1)
data['price'].plot(kind='kde',grid=True, style='-k',title='Density Curve')

# 异常值处理
import numpy as np
new_data = data.copy()
for col in data.columns:
    # if data[col].dtype() == str: continue
    ser = data[col]
    mean = data[col].mean()
    std = data[col].std()
    # new_data[col] = ser[np.abs(ser - mean) <= 3 *std]
    
    ser_c =  ser [ np.abs(ser - mean) <= 3 *std]
    ser_e =  ser [ np.abs(ser - mean) > 3 *std]
    new_data[col] = ser_c
data = new_data.dropna()


##################################################################################################
# 特征间的相关性分析


correlation_matrix = data.corr()

import seaborn as sb
f, ax = plt.subplots(figsize=(9, 6))
ax = sb.heatmap(correlation_matrix,annot = True , fmt ='.1f')
f.tight_layout()
plt.show()


#####################################################################################################
# price标准化

scaler = pre.StandardScaler()
col = 'price'
data[col] = scaler.fit_transform(data[[col]])



######################################################################################################
# price离散化

data_scaled = data.copy()
k = 10
bin = [1.0*i/k for i in range(k + 1) ] 
bin = data.price.describe(percentiles=bin)[4:4+k+1]
bin[0] = bin[0] * (1-1e-10)
data_scaled['price_discretized'] = pd.cut(data['price'],bin,labels=range(k))
frec = data_scaled.groupby('price_discretized')['price_discretized'].count()




######################################################################################################
# 7. 找出与price（房价）相关性最高的三个特征

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
X = data_scaled.drop(['price','price_discretized'], axis=1)  # 除去目标特征和标准化后的目标特征

y = data_scaled['price']
selector = SelectKBest(score_func=f_regression, k=3)
selector.fit(X, y)

selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]
print("与房价(price)相关性最高的三个特征是:", selected_features)


