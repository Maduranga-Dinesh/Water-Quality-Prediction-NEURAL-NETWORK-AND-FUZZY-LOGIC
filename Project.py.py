#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("F:\OUSL\2022\EEX 7241 Neural Network & Fuzzy\mini project\LSTM-model-for-Water-Quality-prediction-master\Water Prediction model using LSTM nn\Data for water Quality Project.csv")


# In[3]:


import os
os.getcwd()


# In[4]:


#chage the directory
os.chdir("F:\\OUSL\\2022\EEX 7241 Neural Network & Fuzzy\\mini project\\LSTM-model-for-Water-Quality-prediction-master\\Water Prediction model using LSTM nn")


# In[5]:


import os
os.getcwd()


# In[6]:


#import data set
import pandas


# In[7]:


dataset = pandas.read_excel("Data for water Quality Project.xlsx")


# In[8]:


dataset


# In[9]:


dataset.head(5)


# In[10]:


dataset.tail(5)


# In[11]:


#Liniear interpolate method use for handle the missing values (NaN values)


# In[12]:


new_dataset = dataset.interpolate


# In[13]:


new_dataset


# In[14]:


new_dataset = dataset.fillna(method="bfill")


# In[15]:


new_dataset


# In[16]:


new_dataset = dataset.fillna(method="ffill",limit=2)


# In[17]:


new_dataset


# In[18]:


new_dataset = dataset.fillna(0)


# In[19]:


new_dataset


# In[20]:


import numpy as np


# In[21]:


#Implement RNN,LSTM using Keras Library


# In[22]:


#Tensorflow part


# In[23]:


import numpy as np


# In[24]:


import tensorflow as tf


# In[25]:


from tensorflow.keras.models import Sequential


# In[26]:


from tensorflow.keras.layers import LSTM,Dense


# In[27]:


from tensorflow.keras import datasets


# In[28]:


import numpy as np


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


# In[32]:


from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# In[33]:


df = pd.read_excel("Data for water Quality Project.xlsx",usecols=['Date','Temp','Rain','NO3','PH'])


# In[34]:


df.head(10)


# In[35]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Date'], drop=True)
df.head(10)


# In[36]:


plt.figure(figsize=(15, 6))
df['PH'].plot();


# In[37]:


plt.figure(figsize=(15, 6))
df['NO3'].plot();


# In[38]:


split_date = pd.Timestamp('2016-12-01')
df =  df['PH']
train = df.loc[:split_date]
test = df.loc[split_date:]
plt.figure(figsize=(15, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test']);


# In[39]:


split_date = pd.Timestamp('2012-12-01')
df =  df['PH']
train = df.loc[:split_date]
test = df.loc[split_date:]
plt.figure(figsize=(20, 10))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test']);


# In[40]:


split_date = pd.Timestamp('2019-05-01')
df =  df['NO3']
train = df.loc[:split_date]
test = df.loc[split_date:]
plt.figure(figsize=(15, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test']);


# In[41]:


split_date = pd.Timestamp('2016-12-01')
df =  df['PH']
train = df.loc[:split_date]
test = df.loc[split_date:]
plt.figure(figsize=(15, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test']);


# In[42]:


scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)


# In[43]:


X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]


# In[44]:


nn_model = Sequential()
nn_model.add(Dense(10, input_dim=1, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)


# In[45]:


df.head(10)


# In[46]:


plt.figure(figsize=(15,6))
plt.scatter(X_test,y_test,color='black')
plt.plot(X_test,y_pred_test_nn,color='blue',linewidth=1)
plt.xticks()
plt.yticks()
plt.show()


# In[47]:


plt.figure(figsize=(15, 6))
df['pH'].plot();


# In[48]:


plt.figure(figsize=(15, 6))
df['PH'].plot();


# In[49]:


df = pd.read_excel("Data for water Quality Project.xlsx",usecols=['Date','Temp','Rain','NO3','PH'])


# In[50]:


plt.figure(figsize=(15, 6))
df['pH'].plot();


# In[51]:


plt.figure(figsize=(15, 6))
df['PH'].plot();


# In[52]:


plt.figure(figsize=(15, 6))
df['NO3'].plot();


# In[53]:


scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)


# In[54]:


df.head(10)


# In[55]:


plt.figure(figsize=(15, 6))
df['PH'].plot();


# In[56]:


data = [[i for i in range(100)]]


# In[57]:


data = np.array(data,dtype=float)


# In[58]:


target = [[i for i in range(1,101)]]


# In[59]:


target = np.array(target,dtype=float)


# In[60]:


data = data.reshape(1,1,100)


# In[61]:


x_test=[i for i in range(1,100)]


# In[62]:


x_test=np.array(x_test).reshape((1,1,100));


# In[63]:


x_test=np.array(x_test).reshape((1,1,100))


# In[64]:


x_test=np.array(x_test).reshape(1,1,100)


# In[65]:


x_test=np.array(x_test).reshape((1,1,100))


# In[66]:


y_test=[i for i in range(101,201)]


# In[67]:


model = Sequential()


# In[68]:


model.add(LSTM(100,input_shape=(1,100),return_sequences=True))


# In[69]:


model.add(Dense(100))


# In[70]:


model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])


# In[71]:


model.fit(data,target, nb_epoch=10000, batch_size=1,verbose=2,validation_data=(x_test,y_test))


# In[72]:


predict = model.predict(x_test)


# In[73]:


predict = model.predict(data)


# In[74]:


predict = model.predict(x_test)


# In[ ]:




