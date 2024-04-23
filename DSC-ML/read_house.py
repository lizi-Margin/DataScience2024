def read_house():
    # 读取数据集
    import pandas as pd

    data = (pd.read_csv('src/house_train.csv')).drop(['id','built_date'],axis=1)
    


    ######################################################################################################
    # 缺失值的检测与处理
    means = data.mean()
    data = data.fillna(means)


    #################################################################################
    # 特征编码

    import  sklearn.preprocessing as pre
    feature_encoder_floor = pre.LabelEncoder()
    floor =data['floor']
    feature_encoder_floor.fit(floor)
    data['floor'] = feature_encoder_floor.transform(floor)




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


    return data

