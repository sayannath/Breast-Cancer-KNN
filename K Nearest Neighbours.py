#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


# In[3]:


dataset = {'k' : [[1,2],[2,3],[3,1]],
           'r': [[6,5],[7,7],[8,6]]} #For any new dataset, We have to bring in this format for using our algorithm

new_features = [3,5] #new point which needs to be classified


# In[4]:


np.sum((np.array([3,3]) - np.array([3,4]))**2)**0.5 #distance formula


# In[5]:


np.linalg.norm(np.array([3,3]) - np.array([3,4])) #Distance formula using np.linalg (linear algebra class)


# In[6]:


distances = [] #List to append distances in 
k = 3 #Value of k in KNN
for group in dataset:#r,k
    for features in dataset[group]: #Single datapoint from dataset is saved in features
        distance = np.linalg.norm(np.array(new_features) - np.array(features)) #distance of new_pt from the datapoint
        distances.append([distance,group]) #distance is appended with group so that we can classify later
#votes = []
#for i in sorted(distances)[:3]:
#    votes.append(i[1])
votes = [i[1] for i in sorted(distances)[:3]]
print(votes)


# In[14]:


Counter(votes).most_common(1) #remove square bracket one by one to see all outputs
#used for finding final classification


# In[ ]:





# In[17]:


def knn(data,predict,k=3): #We define function so that we can reuse it afterwards for new dataset
    distances = []
    for group in data:#r
        for features in data[group]:
            distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([distance,group])
    votes = [i[1] for i in sorted(distances)[0:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


# In[74]:


votes = ['r','r','k']
Counter(votes).most_common(1)[0][0]


# In[60]:


result = knn(dataset,new_features,3) #function call
print(result)


# In[20]:


for group in dataset:
    for features in dataset[group]:
        print(group,features)


# In[21]:


plt.xlabel('X1') #Gives the X axis a label X1
plt.ylabel('X2') #Gives the Y axis a label X2

for group in dataset:
    for features in dataset[group]:
        print(features)
        plt.scatter(features[0],features[1],s=100,color = group) #A single point is plotted on graph
plt.scatter(new_features[0],new_features[1],s=100) #Point to be predicted is plotted on graph
plt.show()


# In[25]:


dataset= {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[8,6],[7,7]]}
new_pt = [3,4]
k=4


# In[30]:


knn(dataset,new_pt,3)


# In[31]:


df = pd.read_csv('Breast-Cancer.csv') #Used to read comma seperated value file and store it in pandas dataframe
df.replace('?',-99999,inplace = True) #replacing missing values denoted by '?' with -99999 in df variable itself
df.drop(['id'],axis=1,inplace=True) #removing the column (axis = 1 means column) 'id' as it is not needed for processing
full_data = df.astype(float).values.tolist() #Converting dataframe into a normal list for further processing
full_data


# In[32]:


len(full_data)


# In[33]:


#THIS PROCESS IS CALLED TRAIN-TEST-SPLIT
x=[1,2,3,4,5,6,7,8,9,10]
test_size = 0.2 #20% data is used as test data
train_data = x[:-int(test_size*len(x))] #Smartly using slicing operation to create train and test data
test_data = x[-int(test_size*len(x)):]
test_data


# In[34]:


int(0.8*699)


# In[76]:


import random
random.shuffle(full_data) #We shuffle our data so that we get some uniformity in the examples used as train and test data
#That is to say that we shouldn't have all examples of only a single class in the test data
#TRAIN-TEST-SPLIT
test_size = 0.2

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
len(train_data) #each list inside main list has 10 elements i.e. 9 features and 1 label, same for test data


# In[77]:


len(full_data)


# In[36]:


train_set = {2:[] , 4:[]} #We need our dataset in dictionary format so that we can use it in KNN function
test_set = {2:[] , 4:[]}
for i in train_data:
    train_set[i[-1]].append(i[:-1])
    #We took one-one list from train_data and appended the features from that list into dictionary 
    #with key as last element (label) of that list
for i in test_data:
    test_set[i[-1]].append(i[:-1]) #same process repeated for test data
train_set


# In[37]:


knn(train_set,[5,10,10,5,4,5,4,4,1],11) #passing a single data [5,10,10,5,4,5,4,4,1] for prediction 
#if tumor is benign (2) or malignant (4)


# In[78]:


accuracy = [] #list to append accuracy in
z = [i for i in range(1,100,2)] #list of all values of k used below
for k in range(1,100,2):
    correct = 0
    total = len(test_data)
    for group in test_set:
        for data in test_set[group]: #data contains a list of 9 features x1,x2...x9
            vote = knn(train_set,data,k) #we check classification made by knn for data variable
            if group == vote: #if group of data variable = classification made by knn then correct += 1
                correct+=1
    accuracy.append(correct/total)


# In[82]:


z[accuracy.index(max(accuracy))]


# In[46]:


#Accuracy list will have some accuracy at 0th index for k =1, at 1st index for k =2 and at nth index for k = n+1
print(max(accuracy),z[accuracy.index(max(accuracy))]) #printing maximum accuracy and first value of kfor which it occurs
#PLOTTING ACCURACY VS. K graph for analysis of performance of KNN on breast cancer classification problem
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.plot(z,accuracy) 
#Plot functions gives a continuous line over all points in z (list of x coordinate) and accuracy (list of y coordinate)
plt.show() #shows the plot


# In[ ]:




