#!/usr/bin/env python
# coding: utf-8

# In[531]:


import pandas as pd
import numpy as np
import math
import random


# In[532]:


data = pd.read_csv('customer_data.csv')
data = np.array(data)
np.random.shuffle(data)


# In[533]:


x1 = data[:,0]
x2 =data[:,1]
y = data[:,2]
x1 = x1.astype(np.float32)
x2 = x2.astype(np.float32)


# In[534]:


x1_min = 100.0
x1_max = 0.0
for i in x1:
    if x1_min > i:
        x1_min =i
    if x1_max < i:
        x1_max = i
dif = (x1_max - x1_min)        
for i in range(400):
    x1[i] = (float(x1[i]-x1_min)/dif)


# In[535]:


x2_min = 10000000.0
x2_max = 0.0
for i in x2:
    if x2_min > i:
        x2_min =i
    if x2_max < i:
        x2_max = i
dif = (x2_max - x2_min)        
for i in range(400):
    x2[i] = (float(x2[i]-x2_min)/dif)    


# In[536]:


x1_train = x1[:350]
x2_train = x2[:350]
y_train = y[:350]
x1_test = x1[350:]
x2_test = x2[350:]
y_test = y[350:]
m = len(x1_train)


# In[537]:


def hypothesis(b,w1,w2,index):
    z = b +(w1*x1_train[index]) + (w2*x2_train[index])
    h = (1.0/(1+np.exp(-z)))
    return h


# In[538]:


def calculate_Jb(b,w1,w2):
    j = 0
    for i in range(m):
        h = ((hypothesis(b,w1,w2,i)) - y_train[i])
        j = j + h 
    j = j / m  
    return j    


# In[539]:


def calculate_Jw1(b,w1,w2):
    j = 0
    for i in range(m):
        h = (((hypothesis(b,w1,w2,i)) - y_train[i])* x1_train[i])
        j = j + h    
    j = j / m   
    return j  


# In[540]:


def calculate_Jw2(b,w1,w2):
    j = 0
    for i in range(m):
        h = (((hypothesis(b,w1,w2,i)) - y_train[i])* x2_train[i])
        j = j + h    
    j = j / m
    return j  


# In[541]:


def cost(b,w1,w2):
    ans = 0
    for i in range (m):
        h = hypothesis(b,w1,w2,i)
        if y_train[i]==1:
            ans = ans + (math.log2(h))
        else:
            ans = ans +(math.log2(1-h))
    ans = ans * (-1/m)    
    return ans


# In[542]:


alpha = 0.1
list1= [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
def gradentDescent():
    b = random.choice(list1)
    w1 =random.choice(list1)
    w2 = random.choice(list1)
    cost_value = cost(b,w1,w2)
    while 1:
        last_value = cost_value
        
        temp_b = b - (alpha * calculate_Jb(b,w1,w2))
        temp_w1 = w1 - (alpha * calculate_Jw1(b,w1,w2))
        temp_w2 = w2 - (alpha * calculate_Jw2(b,w1,w2))
        
        cost_value = cost(temp_b,temp_w1,temp_w2)
        
        cost_value = round(cost_value,4)
        
        if cost_value >= last_value:
            break
        b = temp_b
        w1 = temp_w1
        w2 = temp_w2
    return b,w1,w2


# In[543]:


b,w1,w2 = gradentDescent()


# In[544]:


def hypothesis_test(b,w1,w2,index):
    z = b +(w1*x1_test[index]) + (w2*x2_test[index])
    h = (1.0/(1+(pow(math.e,(-1*z)))))
    return h


# In[545]:


accuracy =0.0
for i in range(0,50):
    h = hypothesis_test(b,w1,w2,i)
    y = 0
    if h >= 0.5:
        y = 1
    else:
        y= 0
    if y==y_test[i]:
        accuracy = accuracy + 1
accuracy = (accuracy / 50)*100
print(accuracy)


# In[ ]:




