#%% Dataseti yükleme ve tanımlama
from sklearn import datasets
dataSet=datasets.load_iris()
features=dataSet.data
labels=dataSet.target
labelnames=list(dataSet.target_names)
featuresNames=dataSet.feature_names
print(featuresNames)
print( [labelnames[i] for i in labels[:3]])
#%%

#%% Veri analiz
import pandas as pd
print(type(features))
featuresDF=pd.DataFrame(features)
featuresDF.columns=featuresNames
print(featuresDF.info())
#%%
featuresDF.plot(kind='box')

#%% Model seçimi

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

#%% model eğitimi
#split
import numpy as np
from sklearn.model_selection import train_test_split

X=features
y=labels

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)



#%%

#%%Fit etmek,eğitim

model.fit(X_train,y_train)



#%%

#%%Test
accuracy=model.score(x_test,y_test)
print('data test doğruluk oranı {:.2}%'.format(accuracy))

#%%

#%%save model
from joblib import dump,load
filename='my_second_saved_model'

dump(model,filename)

#%%

#%%laod_model
model=load('my_second_saved_model')

