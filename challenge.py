
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[2]:


train=pd.read_csv('train.csv')


# In[3]:


test=pd.read_csv('test.csv')


# In[4]:


train.head()


# train.dtypes

# # some data preprocessing to bring damage_grade and building_id in respective from

# In[5]:


from sklearn import preprocessing


# In[6]:


le=preprocessing.LabelEncoder()


# In[7]:


le.fit(train['damage_grade'])


# In[8]:


le.fit(train['building_id'])


# In[9]:


list(le.classes_)


# In[10]:


train['building_id']=le.transform(train['building_id'])


# In[11]:


train.head()


# In[12]:


lt=preprocessing.LabelEncoder()


# In[13]:


lt.fit(train['damage_grade'])


# In[14]:


list(lt.classes_)


# In[15]:


train['damage_grade']=lt.transform(train['damage_grade'])


# In[16]:


train.head()


# # preprocessing of area assesed

# In[17]:


lk=preprocessing.LabelEncoder()


# In[18]:


lk.fit(train['area_assesed'])


# In[19]:


train['area_assesed']=lk.transform(train['area_assesed'])


# In[20]:


train.head()


# # spliting dependant and independant variables

# In[21]:


X=train.drop(['damage_grade'], axis=1)


# In[22]:


Y=train['damage_grade']


# # fitting dataset into models

# In[23]:


X.fillna(0)


# In[24]:


Y.fillna(0)


# In[25]:


X['has_geotechnical_risk']=X['has_geotechnical_risk'].astype('int64')


# In[26]:


X.drop(['has_repair_started'],axis=1)


# In[34]:


X_train=X.drop(['has_repair_started'],axis=1)
X_train


# # importing test dataset

# In[35]:


test=pd.read_csv('test.csv')


# # preprocessing of test dataset

# In[36]:


l1=preprocessing.LabelEncoder()
l2=preprocessing.LabelEncoder()
l1.fit(test['building_id'])
l2.fit(test['area_assesed'])
test['area_assesed']=l2.transform(test['area_assesed'])
test['building_id']=l1.transform(test['building_id'])


# In[37]:


X_test=test


# In[38]:


test.dtypes


# In[39]:


X_test['has_geotechnical_risk']=test['has_geotechnical_risk'].astype('int64')


# In[40]:


X_test.fillna(0)


# In[41]:


X_test.drop(['has_repair_started'], axis=1)


# In[215]:


X_test


# In[216]:


X.dtypes


# In[42]:


Y.dtypes


# In[44]:


from sklearn import  linear_model


# # implementing multiclass classification

# In[45]:


mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(X_train,Y)


# In[46]:


X_test.fillna(0)
X2=X_test.drop(['has_repair_started'], axis=1)


# In[47]:


X2


# In[48]:


y_pred = mul_lr.predict(X2)


# In[49]:


Y=list(y_pred)
X2['damage_grade']=Y

X2=X2.sort_values("damage_grade",ascending=False)


# In[50]:


X2


# In[149]:


X_test[X_test['damage_grade']==3]


# # time to retransform damage_grade and building_id

# In[56]:


X2['building_id']=l1.inverse_transform(X2['building_id'])


# In[57]:


X2


# In[58]:


X2['damage_grade']=lt.inverse_transform(X2['damage_grade'])


# In[59]:


X2


# In[67]:


final_csv1=X2[['building_id','damage_grade']]


# In[68]:


final_csv1.to_csv('final_result1.csv',index = False)


# In[72]:


building_structure=pd.read_csv('building_structure.csv')


# In[73]:


BO=pd.read_csv('Building_Ownership_Use.csv')


# In[74]:


train1=pd.read_csv('train.csv')
test1=pd.read_csv('test.csv')


# In[75]:


result = pd.merge(building_structure,BO, on='building_id')
res_train=pd.merge(train1,result,on="building_id")
res_test=pd.merge(test1,result,on="building_id")


# In[76]:


res_train.fillna(0)


# In[77]:


X_res_train=res_train.drop(['damage_grade'],axis=1)
X_res_train['building_id']=le.transform(res_train['building_id'])


# In[78]:


X_res_train['area_assesed']=lk.transform(res_train['area_assesed'])


# In[80]:


X_res_train.fillna(0)


# In[83]:


l=preprocessing.LabelEncoder()
Y_res_train=res_train['damage_grade']
l.fit(res_train['damage_grade'])
Y_res_train=l.transform(res_train['damage_grade'])


# In[84]:


multi_lr = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(X_res_train,Y_res_train)


# In[87]:


X_res_train.dtypes


# In[92]:


X=X_res_train.drop(['condition_post_eq','land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration'],axis=1)


# In[96]:


X


# In[101]:


owner_stats=preprocessing.LabelEncoder()
owner_stats.fit(X['legal_ownership_status'])
X['legal_ownership_status']=owner_stats.transform(X['legal_ownership_status'])


# In[102]:


X.dtypes


# In[114]:


del X['has_repair_started']


# In[107]:


multi_lr = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(X,Y_res_train)


# In[61]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[62]:


BO=pd.read_csv('Building_Ownership_Use.csv')
BS=pd.read_csv('Building_Structure.csv')
data=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')


# In[63]:


merged = pd.merge(BO, BS, on = 'building_id')
merged_data = pd.merge(left = data, right = merged, how = 'left', on= 'building_id')
merged_test = pd.merge(left = testdata, right = merged, how = 'left', on= 'building_id')


# In[64]:


merged_data.dtypes


# In[65]:


merged_data.fillna(0)


# In[66]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
lt=preprocessing.LabelEncoder()
lt.fit(merged_data['damage_grade'])
le.fit(merged_data['building_id'])
merged_data['damage_grade']=lt.transform(merged_data['damage_grade'])
lk=preprocessing.LabelEncoder()
lk.fit(merged_data['area_assesed'])
merged_data['area_assesed']=lk.transform(merged_data['area_assesed'])


# In[67]:


X_Train=merged_data.drop(['damage_grade'],axis=1)


# In[68]:


X_Train=X=merged_data.drop(['condition_post_eq','land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration'],axis=1)


# In[69]:


X_Train['building_id']=le.transform(merged_data['building_id'])


# In[71]:


X_Train.dtypes


# In[72]:


owner_stats=preprocessing.LabelEncoder()
owner_stats.fit(X_Train['legal_ownership_status'])
X_Train['legal_ownership_status']=owner_stats.transform(X_Train['legal_ownership_status'])
X_Train['has_geotechnical_risk']=X_Train['has_geotechnical_risk'].astype('int64')
#X_Train['count_families']=X_Train['count_families'].astype('int64')
X_Train['has_secondary_use']=X_Train['has_secondary_use'].astype('int64')


# In[73]:


X_Train['count_families']=X_Train['count_families'].fillna(0).astype(int)


# In[74]:


X_Train['has_repair_started']=X_Train['has_repair_started'].fillna(0).astype(int)


# In[75]:


X_Train.dtypes


# In[76]:


Y_Train=merged_data['damage_grade']


# In[77]:


Y_Train


# In[78]:


X_Train.fillna(0)
Y_Train.fillna(0)


# In[79]:


X=X_Train.fillna(0).drop(['damage_grade'],axis=1)
Y=Y_Train.fillna(0)


# In[94]:


X_Train.dtypes


# In[95]:


from sklearn import  linear_model
mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(X_Train,Y)


# In[28]:


X_Test=merged_test.fillna(0)


# In[33]:


X_Test


# In[34]:


from sklearn import preprocessing
le.fit(X_Test['building_id'])
lk.fit(X_Test['area_assesed'])
X_Test['area_assesed']=lk.transform(X_Test['area_assesed'])
X_Test['building_id']=le.transform(X_Test['building_id'])
owner_stats.fit(X_Test['legal_ownership_status'])
X_Test['legal_ownership_status']=owner_stats.transform(X_Test['legal_ownership_status'])


# In[35]:


X_Test=X=X_Test.drop(['condition_post_eq','land_surface_condition','foundation_type','roof_type','ground_floor_type','other_floor_type','position','plan_configuration'],axis=1)


# In[37]:


X_Test.dtypes


# In[38]:


X_Test['has_geotechnical_risk']=X_Test['has_geotechnical_risk'].astype('int64')
X_Test['has_secondary_use']=X_Test['has_secondary_use'].astype('int64')
X_Test['count_families']=X_Test['count_families'].fillna(0).astype(int)
X_Test['has_repair_started']=X_Test['has_repair_started'].fillna(0).astype(int)


# In[105]:


X_Test.dtypes
y_pred = mul_lr.predict(X_Test)
Y=list(y_pred)
X_Test['damage_grade']=Y

X_Test=X_Test.sort_values("damage_grade",ascending=False)


# In[106]:


X_Test


# In[99]:


X_Train.dtypes


# In[47]:


X


# In[108]:


X_Test['building_id']=le.inverse_transform(X_Test['building_id'])
X_Test['damage_grade']=lt.inverse_transform(X_Test['damage_grade'])


# In[111]:


final_csv1=X_Test[['building_id','damage_grade']]

final_csv1.to_csv('final_result_2.csv',index = False)

