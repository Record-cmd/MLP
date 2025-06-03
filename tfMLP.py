import numpy as np
import tensorflow as tf
import sys, os
sys.path.append(os.pardir)
import pandas as pd
import functions as fs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

scaler = MinMaxScaler()
early= EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
smote = SMOTE()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

diabetes_frame = fs.read_Data()                                                                 #sparse_output=False << numpy array반환

x = diabetes_frame.drop('Diabetes_012', axis = 1) #목표변수를 제외한 속성들
y = diabetes_frame['Diabetes_012'] #목표변수

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) #테스트 데이터를 0.8로 나눔

y_train = y_train.astype(int)
y_test  = y_test.astype(int)


print('='*20)
print("y_train 비율")
print(y_train.value_counts(normalize=True))

print("y_test 비율")
print(y_test.value_counts(normalize=True))
print('='*20)
print(f"y_train의 value_counts : \n {y_train.value_counts()}")
print(f"y_test의 value_counts : \n {y_test.value_counts()}")
print('='*20)

x_train_resampled,y_train=smote.fit_resample(x_train, y_train)

x_train = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)

#가중치계산
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("클래스 가중치:", class_weight_dict)


#y_train_enc = tf.one_hot(y_train, depth=3).numpy()
#y_test_enc = tf.one_hot(y_test,  depth=3).numpy()

#print("y_train 원핫코딩 결과")
#print(y_train_enc)
#print("y_test 원핫코딩 결과")
#print(y_test_enc)




n_input = 21
n_hidden = 100
n_output = 3

mlp = Sequential()
mlp.add(Dense(units = n_hidden,activation = 'relu', input_shape = (n_input,),kernel_initializer = 'random_uniform' , bias_initializer='zeros'))
mlp.add(Dense(units = n_hidden,activation = 'relu',kernel_initializer = 'random_uniform' , bias_initializer='zeros'))
mlp.add(Dense(units = n_hidden,activation = 'relu',kernel_initializer = 'random_uniform' , bias_initializer='zeros'))

mlp.add(Dense(units=n_output, activation = 'softmax',kernel_initializer = 'random_uniform' , bias_initializer='zeros'))


mlp.compile(loss = "sparse_categorical_crossentropy", optimizer = Adam(learning_rate = 0.001),metrics=['accuracy'])

hist = mlp.fit(x_train,y_train, batch_size=128, epochs=50, validation_data=(x_test,y_test),verbose=2, class_weight=class_weight_dict, callbacks=[early])

res = mlp.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)


y_pred_sample = mlp.predict(x_test)
y_pred_classes_sample = np.argmax(y_pred_sample, axis=1)

report = classification_report(y_test, y_pred_classes_sample, digits=4)
print(report)


import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Train','Validation'],loc='upper left')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Validation'],loc='upper right')
plt.grid()
plt.show()

