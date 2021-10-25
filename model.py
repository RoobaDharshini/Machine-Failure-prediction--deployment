import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('Maintenance.csv')

print(df.head)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Type']=le.fit_transform(df['Type'])
df=df.drop('Rotational speed [rpm]',axis=1)

x=df.iloc[:,2:7]
y=df['Machine failure']

min_max=MinMaxScaler()
x=min_max.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(y_train.shape)

rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)

import pickle

pickle.dump(rf_model,open('model.pkl','wb'))