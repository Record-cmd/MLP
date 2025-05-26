import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
#pd.set_option('display.max_columns', None) #컬럼 ... 생략 여부 설정


def read_Data():
    diabetes_binary = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv') #목표클래스 0,1,2
    #diabetes_binary = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv') #목표 클래스 0,1
    #diabetes_binary = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    return diabetes_binary

def Balance_Calculation(DataSet, column_name): #클래스불균형 출력
    class_distribution = DataSet[column_name].value_counts(normalize=True)
    return class_distribution

def Column_Ratio(DataSet, Group_column_name ,column_name): #컬럼당 당뇨병 비율 리턴
    return DataSet.groupby(Group_column_name)[column_name].value_counts(normalize=True).unstack()

def Column_Value_Count(DataSet, column_name):
    return DataSet[column_name].value_counts()

