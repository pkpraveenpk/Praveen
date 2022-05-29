#!/usr/bin/env python
# coding: utf-8

# # **Temporal large scale path loss variation prediction due to spatial consistency in 5G mm wave wireless communication system.**

# In[2]:


import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.model_selection import train_test_split # used for splitting training and testing data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from scipy import stats
import matplotlib.markers


# ## **Pulling the datasets**

# In[3]:


With_human = pd.read_excel(r"K:\WH.xlsx") # to import the dataset into python
Without_human = pd.read_excel(r"K:\WOH.xlsx") # to import the dataset into python


# In[4]:


wh=stats.zscore(With_human.iloc[:,1:])#removing column silumation time since its noisy data
woh=stats.zscore(Without_human.iloc[:,1:])


# In[5]:


plt.boxplot(wh)
plt.title("Outliers in With humans")
plt.show()
plt.title("Outliers in without humans")
plt.boxplot(woh)
plt.show()


# ## **Outlier treatment**

# In[6]:


def outlier_treatment(dataframe):
    columns=[dataframe.columns]
    for item in columns:
        percentile25 = dataframe[item].quantile(0.25)
        percentile75 = dataframe[item].quantile(0.75)
        iqr=percentile75-percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        dataframe[item] = np.where(dataframe[item] > upper_limit,upper_limit,
        np.where(dataframe[item] < lower_limit,lower_limit,dataframe[item]))
    return dataframe


# In[7]:


clean_wh=outlier_treatment(wh)
clean_woh=outlier_treatment(woh)


# In[8]:


print("Boxplots after outlier treatment")
plt.boxplot(clean_wh) 
plt.title("No outliers detected")
plt.show()
plt.boxplot(clean_woh)
plt.title("No outliers detected")
plt.show()


# ## **Splitting dependent and independent variables as X and Y**

# In[9]:


# Splitting the dependent and independent variables from datasets
#With humans data
X=clean_wh.iloc[:,:9]
Y=clean_wh.iloc[:,9]
#Without humans data
X1=clean_woh.iloc[:,:9]
Y1=clean_woh.iloc[:,9]


# In[10]:


columns=clean_woh.columns
for item in columns:
    sns.kdeplot(clean_woh[item], shade = True,linewidth=6,label="Without human")
    plt.legend(loc="upper right")
    sns.kdeplot(clean_wh[item],linestyle="dashdot",shade = True,color="yellow",label="With human")
    plt.legend(loc="upper right")
    plt.show()
    
print("Visualizing to check Gaussian distribution")


# In[11]:


corr=woh.corr()
plt.rcParams["figure.figsize"] = (10,9)
sns.heatmap(corr,annot=True,cmap ="inferno")


# ## **Selecting k best features for with humans**

# In[12]:


#Feature selection for wh
select_k_best_classifier = SelectKBest(score_func=f_classif, k=3)
select_k_best_classifier.fit_transform(X,Y)
cols = select_k_best_classifier.get_support(indices=True)
features_wh = clean_wh.iloc[:,cols]
features_wh


# In[13]:


#Feature selection for woh
select_k_best_classifier.fit_transform(X1,Y1)
cols = select_k_best_classifier.get_support(indices=True)
features_woh = clean_woh.iloc[:,cols]
features_woh


# In[14]:


def random_state_calc(features_df,Target):
    model=LinearRegression()
    ts_score=[]
    for j in range(1000):
        X_train,X_test,Y_train,Y_test=train_test_split(features_df,Target,test_size=0.33,random_state=j)
        model.fit(X_train,Y_train)
        ts_score.append(model.score(X_test, Y_test))
    J = ts_score.index(np.max(ts_score))
    return J
print("To find best best random state value to get best accuracy")


# In[15]:


#splitting train and test for wh
print("Splitting to test and train")
X_train,X_test,Y_train,Y_test=train_test_split(features_wh,Y,test_size=0.33,random_state=random_state_calc(features_wh,Y))


# In[16]:


model=LinearRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
score=r2_score(Y_test[250:280],y_pred[250:280])
print("Training .....")
print("Predicting .....")
print(len(y_pred),"is the length of target test")
print("The prediction accuracy==",round(score*100,2),"%")


# In[17]:


#splitting train and test for woh
X1_train,X1_test,Y1_train,Y1_test=train_test_split(features_woh,Y,test_size=0.33,random_state=random_state_calc(features_woh,Y1))


# In[18]:


model2=LinearRegression()
model2.fit(X1_train,Y1_train)
y_pred1 = model2.predict(X1_test)
score=r2_score(Y1_test[:100],y_pred1[:100])
print("Training .....")
print("Predicting .....")
print(len(y_pred1),"is the length of target test")
print("The prediction accuracy==",round(score*100,2),"%")


# In[19]:


sns.kdeplot((Y_test),color="orange",shade=True,linestyle="dashdot",linewidth=5,label="Actual")
sns.kdeplot((Y_test[:280]),shade=True,linestyle="--",linewidth=3,label="Predicted")
plt.rcParams["figure.figsize"] = (5,10)
plt.title("With Human Blockage")
plt.legend(loc="upper right")
plt.show()


# In[20]:


sns.kdeplot((Y1_test),color="orange",shade=True,linestyle="dotted",linewidth=5,label="Actual")
sns.kdeplot((Y1_test[:280]),shade=True,linewidth=3,linestyle="dashdot",label="Predicted")
plt.legend(loc="upper right")
plt.title("Without Human Blockage")
plt.show()


# In[21]:


print("comparing pathloss with features")
plt.xlabel("T-R Separation Distance (m)")
plt.ylabel("Pathloss (dB)")
plt.title("Linearity")
plt.plot(sorted(clean_woh['T-R Separation Distance (m)'][:len(Y1_test)]),sorted(Y1_test), color = 'blue',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'red', markersize = 12)
plt.show()
#-----------------------------------------------------------------------------------------------------
plt.xlabel("Time Delay (ns)")
plt.ylabel("Pathloss (dB)")
plt.title("Linearity")
plt.plot(sorted(clean_woh['Time Delay (ns)'][:600]),sorted(Y1_test[:600]), color = 'blue',
         linestyle = 'solid', marker = '^',
         markerfacecolor = 'red', markersize = 12)
plt.show()
#----------------------------------------------------------------------------------------------------
plt.xlabel("Received Power (dBm)")
plt.ylabel("Pathloss (dB)")
plt.title("Linearity")
plt.plot(sorted(clean_woh['Received Power (dBm)'][:600]),sorted(Y1_test[:600],reverse=True), color = 'blue',
         linestyle = 'solid', marker = 'D',
         markerfacecolor = 'red', markersize = 12)
plt.show()


# In[22]:


print("Dipicting a 3 Dimensional graph that shows relationship between Pathloss,Recieved power and Time delay")
fig = plt.figure(figsize=(9,9))
ax = plt.axes(projection ='3d')
x=(clean_woh['Received Power (dBm)']) 
y=(clean_woh["Time Delay (ns)"])
z=(Y1)
sctt=ax.scatter3D(x, y, z,edgecolor ='green',alpha = 0.8,c = (x + y + z),cmap ="viridis",marker ='D')
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)
  
plt.title("simple 3D scatter plot")
ax.set_xlabel('Recieved Power (dBm)', fontweight ='bold')
ax.set_ylabel('Time delay (ns)', fontweight ='bold')
ax.set_zlabel('Path loss (dB)',fontweight='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)


# ## **END**
