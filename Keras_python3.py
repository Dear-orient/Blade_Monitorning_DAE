# __author__ = 'orient'
# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import warnings
from keras.models import Sequential,load_model
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.optimizers import SGD
from keras.datasets import mnist

warnings.filterwarnings("ignore")
#测试Python3.5.2
print('hello python3.5.2')
# sess = tf.Session()
# matrixl = tf.constant([[3.,3.]])
# matrix2 = tf.constant([[2.],[2.]])
# print(sess.run(matrixl))
# print(sess.run(matrix2))
# product = tf.matmul(matrixl,matrix2)
# print(sess.run(product))
# sess.close()

def load_samples(dataset="training_data"):
    datas = np.load('Labdata_Normal_Scale_allNew.npy')
    permutation = np.random.permutation(datas.shape[0])
    datas = datas[permutation, :]
    if dataset == "training_data":
        trainData = datas[0:int(datas.shape[0] * 1), :]
        # trainData = [np.reshape(x, (200, 1)) for x in trainData]
    elif dataset == "testing_data":
        testData = datas[int(datas.shape[0] * 0.9):datas.shape[0], :]
        # testData = [np.reshape(x, (200, 1)) for x in testData]
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    if dataset == "training_data":
        pair = trainData
        return pair
    elif dataset == 'testing_data':
        pair = testData
        return pair
    else:
        print('Something wrong')

# 第一步，选择模型
model = Sequential()
model.add(Dense(700,input_shape=(1000,),kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation(LeakyReLU(0.1)))
# model.add(Dropout(0.3))

# model.add(Dense(500))
# model.add(Activation(LeakyReLU(0.1)))

model.add(Dense(1000,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('linear'))
sgd = SGD(lr=0.001,momentum=0.3,decay=1e-6,nesterov = False)
model.compile(loss='mean_squared_error', optimizer='Adam')

time0 = time.time()
train_set = load_samples(dataset='training_data')
# test_set = load_samples(dataset='testing_data')
print('It takes %f seconds to read the data.' % (time.time()-time0))
model.fit(train_set,train_set,batch_size=2500,epochs=100,verbose=2)


# 保存模型
# model.save('LabData_my_modelNew.h5')
# model = load_model('LabData_my_model.h5')
# 获取参数
# for layer in model.layers:
#     weights = layer.get_weights()  # list of numpy array
#     print('***********************')
#     print(weights)

ERROR_train = []
for data in train_set:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_train.append(scores)
np.save('ERROR_train.npy',ERROR_train)
print('save ERROR_train complement.')

datas = np.load('Labdata_Normal_Scale_test.npy')
ERROR_test = []
for data in datas:
    data = np.array([data])
    tempt = data
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_test.append(scores)
np.save('ERROR_test.npy',ERROR_test)
print('save ERROR_test complement.')
print('data:',tempt)
print(model.predict(tempt,batch_size=1))

datas = np.load('Labdata_weight_Scale_add2.npy')
ERROR_add2 = []
for data in datas:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_add2.append(scores)
np.save('ERROR_add2.npy',ERROR_add2)
print('save ERROR_add2 complement.')

datas = np.load('Labdata_weight_Scale_add4.npy')
ERROR_add4 = []
for data in datas:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_add4.append(scores)
np.save('ERROR_add4.npy',ERROR_add4)
print('save ERROR_add4 complement.')

datas = np.load('Labdata_weight_Scale_add5.npy')
ERROR_add5 = []
for data in datas:
    data = np.array([data])
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_add5.append(scores)
np.save('ERROR_add5.npy',ERROR_add5)
print('save ERROR_add5 complement.')

datas = np.load('Labdata_weight_Scale_add5_long.npy')
ERROR_add5_long = []
for data in datas:
    data = np.array([data])
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_add5_long.append(scores)
np.save('ERROR_add5_long.npy',ERROR_add5_long)
print('save ERROR_add5_long complement.')


# 针对不同档的风速/裂纹进行测试
datas = np.load('Labdata_Crackle_Scale_2cm.npy')
ERROR_Crackle_2cm = []
for data in datas:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_Crackle_2cm.append(scores)
np.save('ERROR_Crackle_2cm.npy',ERROR_Crackle_2cm)
print('save ERROR_Crackle_2cm complement.')

datas = np.load('Labdata_Crackle_Scale_3cm.npy')
ERROR_Crackle_3cm = []
for data in datas:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_Crackle_3cm.append(scores)
np.save('ERROR_Crackle_3cm.npy',ERROR_Crackle_3cm)
print('save ERROR_Crackle_3cm complement.')

datas = np.load('Labdata_Crackle_Scale_4cm.npy')
ERROR_Crackle_4cm = []
for data in datas:
    data = np.array([data])
    # print(data)
    scores = model.evaluate(data,data,batch_size=1,verbose=0)
    ERROR_Crackle_4cm.append(scores)
np.save('ERROR_Crackle_4cm.npy',ERROR_Crackle_4cm)
print('save ERROR_Crackle_4cm complement.')