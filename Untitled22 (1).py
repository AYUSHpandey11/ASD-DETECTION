#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\pande\Downloads\Toddler_Autism_dataset_July_2018.csv")


# In[3]:


df.info()


# In[4]:


df = df.rename(columns = {'Class/ASD Traits ' : 'ASD'})
df = df.rename(columns = {'A1' : 'Q1'})
df = df.rename(columns = {'A2' : 'Q2'})
df = df.rename(columns = {'A3' : 'Q3'})
df = df.rename(columns = {'A4' : 'Q4'})
df = df.rename(columns = {'A5' : 'Q5'})
df = df.rename(columns = {'A6' : 'Q6'})
df = df.rename(columns = {'A7' : 'Q7'})
df = df.rename(columns = {'A8' : 'Q8'})
df = df.rename(columns = {'A9' : 'Q9'})
df = df.rename(columns = {'A10' : 'Q10'})
df = df.rename(columns = {'Age_Mons' : 'Age in Months'})
df = df.rename(columns = {'Sex' : 'Gender'})
df = df.rename(columns = {'Qchat-10-Score' : 'Score out of 10'})
df = df.rename(columns = {'Ethnicity' : 'Region'})
df


# In[5]:


x = df.drop('Case_No' , axis = 1)
x = x.drop('ASD', axis = 1)
y = df['ASD']


# In[6]:


x.shape, y.shape


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 36)


# In[8]:


obj_cols = x_train.select_dtypes(include = 'object').columns
obj_cols


# In[9]:


float_cols = x_train.select_dtypes(include = 'int64').columns
float_cols


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(y_train)


# In[11]:


y_train_processed = le.transform(y_train)
y_test_processed = le.transform(y_test)


# In[12]:


y_train_processed


# In[13]:


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories = [x_train[i].unique() for i in obj_cols])
oe.fit(x_train[obj_cols])
x_train_cat_encoded = oe.transform(x_train[obj_cols])
x_test_cat_encoded = oe.transform(x_test[obj_cols])


# In[14]:


x_train_cat_encoded


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train[float_cols])

x_train_float_encoded = scaler.transform(x_train[float_cols])
x_test_float_encoded = scaler.transform(x_test[float_cols])


# In[16]:


x_train_float_encoded


# In[17]:


x_train_processed = np.hstack((x_train_cat_encoded, x_train_float_encoded))
x_test_processed = np.hstack((x_test_cat_encoded, x_test_float_encoded))


# In[18]:


feature_names = np.concatenate([obj_cols, float_cols])


# In[19]:


x_train_final = pd.DataFrame(x_train_processed, columns = feature_names)


# In[20]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_processed, y_train_processed)
y_pred = lr.predict(x_test_processed)
print(accuracy_score(y_test_processed, y_pred))
print(confusion_matrix(y_test_processed, y_pred))


# In[24]:


le.inverse_transform([0,1])


# In[29]:


def pretty_confusion_matrix(y_test, y_pred, labels = ['Not_Diagnosed_with_ASD', 'Diagnosed_with_ASD']):
    cm = confusion_matrix(y_test, y_pred)
    pred_labels = ['Predicted ' + i for i in labels]
    df = pd.DataFrame(cm, columns = pred_labels, index = labels)
    return df


# In[30]:


results_plot = pretty_confusion_matrix(y_test_processed, y_pred, ['Not_Diagnosed_with_ASD', 'Diagnosed_with_ASD'])
results_plot


# In[31]:


sns.heatmap(results_plot, cmap = 'Spectral')


# In[32]:


lr.coef_


# In[33]:


# Match coef's of features to columns
feature_dict = dict(zip(x.columns, list(lr.coef_[0])))
feature_dict


# In[34]:


# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
# plt.style.use('dark_background')
feature_df.T.plot.bar(title="Feature Importance", legend=False)


# In[35]:


yes_autism = df[df['ASD'] == 'Yes']
ax = plt.subplot()
# plt.style.use('dark_background')
sns.countplot(x = "Age in Months", data = yes_autism)
ax.set_ylabel('Number of Children')
ax.set_title('Age Distribution')


# In[36]:


fig = plt.gcf()
plt.pie(x['Gender'].value_counts(), labels = ('Boy','Girl'), explode = [0.1, 0], autopct='%1.1f%%', shadow = True, startangle = 90,
        colors = ['orangered', 'royalblue'])
plt.axis('equal')
plt.show()


# In[38]:


df2 = pd.read_csv(r"C:\Users\pande\Downloads\Sample_Dataset.csv")
df2.info()


# In[26]:


X = df2.drop('Case_No' , axis = 1)
X = X.drop('ASD', axis = 1)
Y = df2['ASD']
X.shape, Y.shape


# In[27]:


obj_cols2 = X.select_dtypes(include = 'object').columns
obj_cols2


# In[28]:


float_cols2 = X.select_dtypes(include = 'int64').columns
float_cols2


# In[234]:


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories = [X[i].unique() for i in obj_cols2])
oe.fit(X[obj_cols2])
X_cat_encoded = oe.transform(X[obj_cols2])


# In[235]:


X_cat_encoded


# In[236]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X[float_cols2])

X_float_encoded = scaler.transform(X[float_cols2])


# In[237]:


X_processed = np.hstack((X_cat_encoded, X_float_encoded))


# In[238]:


feature_names2 = np.concatenate([obj_cols2, float_cols2])


# In[239]:


X_final = pd.DataFrame(X_processed, columns = feature_names2)


# In[240]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_processed, y_train_processed)
y_pred = lr.predict(X_processed)


# In[241]:


print("ASD Possible : ", y_pred)

