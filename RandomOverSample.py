import functions as fs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from two_layer_net import TwoLayerNet
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


diabetes_frame = fs.read_Data()
encoder = OneHotEncoder(sparse_output=False)                                                                            #sparse_output=False << numpy array반환
ros = RandomOverSampler()

x = diabetes_frame.drop('Diabetes_012', axis = 1) #목표변수를 제외한 속성들
y = diabetes_frame['Diabetes_012'] #목표변수

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) #테스트 데이터를 0.8로 나눔

print("y_train 비율")
print(y_train.value_counts(normalize=True))

print("y_test 비율")
print(y_test.value_counts(normalize=True))
print('='*20)

print(f"y_train의 value_counts : \n {y_train.value_counts()}")
print(f"y_test의 value_counts : \n {y_test.value_counts()}")
print('='*20)

x_train_resampled,y_train_resampled=ros.fit_resample(x_train, y_train)

print("y_train 기존 결과")
print(y_train_resampled)
print("y_test 기존 결과")
print(y_test)

x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)

y_train_resampled = encoder.fit_transform(y_train_resampled.values.reshape(-1,1))
y_test_enc = encoder.transform(y_test.values.reshape(-1,1)) #목표변수


print("y_train 원핫코딩 결과")
print(y_train_resampled)
print("y_test_enc 원핫코딩 결과")
print(y_test_enc)

labels = y_train_resampled.argmax(axis=1)

unique, counts = np.unique(labels, return_counts=True)
class_counts = dict(zip(unique, counts))
print("샘플링 이후 각 클래스별 개수")
print(class_counts)


#x_test = x_test.to_numpy()
#x_train_resampled = x_train_resampled.to_numpy()

#--------------------------------------------------------------------------

network = TwoLayerNet(input_size = 21, hidden_size = 100, output_size=3)

iters_num= 25000
train_size = x_train_resampled.shape[0]
print(train_size)
batch_size = 1026
learning_rate=0.01

TrainClass_0 = []
TrainClass_1 = []
TrainClass_2 = []
TrainG_mean = []

TestClass_0 = []
TestClass_1 = []
TestClass_2 = []
TestG_mean = []

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_num = []


iter_per_epoch = int(max(train_size/batch_size,1))
j=1
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train_resampled[batch_mask]
    t_batch = y_train_resampled[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        
        train_acc = network.accuracy(x_train_resampled, y_train_resampled)
        test_acc = network.accuracy(x_test, y_test_enc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        iter_num.append(i)
        print(i,train_acc, test_acc)
        
        train_class_acc, Train_Class_0_acc, Train_Class_1_acc, Train_Class_2_acc, Train_G_Mean = network.accuracy2(x_train_resampled, y_train_resampled)
        test_class_acc, Test_Class_0_acc, Test_Class_1_acc, Test_Class_2_acc, Test_G_Mean = network.accuracy2(x_test, y_test_enc)

        TrainClass_0.append(Train_Class_0_acc)
        TrainClass_1.append(Train_Class_1_acc)
        TrainClass_2.append(Train_Class_2_acc)
        TrainG_mean.append(Train_G_Mean)
        
        TestClass_0.append(Test_Class_0_acc)
        TestClass_1.append(Test_Class_1_acc)
        TestClass_2.append(Test_Class_2_acc)
        TestG_mean.append(Test_G_Mean)
        
        print(f"Epoch {i} - Train Accuracy: {train_class_acc}, Test Accuracy: {test_class_acc}")
        print(f"Epoch {i} - Train G_Mean: {Train_G_Mean}, Test G_mean: {Test_G_Mean}")
        print(f"{j}번 수행")
        j+=1


import matplotlib.pyplot as plt
plt.plot(iter_num, train_acc_list, label = "train")
plt.plot(iter_num, test_acc_list, label = "test")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(iter_num, TrainClass_0, label = "train_0")
plt.plot(iter_num, TestClass_0, label = "test_0")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(iter_num, TrainClass_1, label = "train_1")
plt.plot(iter_num, TestClass_1, label = "test_1")
plt.legend()
plt.show()


import matplotlib.pyplot as plt
plt.plot(iter_num, TrainClass_2, label = "train_2")
plt.plot(iter_num, TestClass_2, label = "test_2")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(iter_num, TrainG_mean, label = "TrainG_mean")
plt.plot(iter_num, TestG_mean, label = "TestG_mean")
plt.legend()
plt.show()

