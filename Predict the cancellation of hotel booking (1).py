#!/usr/bin/env python
# coding: utf-8

# # Predict the cancellation of hotel booking
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('D:\project\hotel_bookings.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[7]:


def data_clean(df):
     df.fillna(0,inplace=True)


# In[8]:


print(df.isnull().sum())


# In[9]:


data_clean(df)


# In[10]:


df.columns


# In[11]:


list=['adults','children','babies']
for i in list:
   print('{} has unique values as {}'.format(i,df[i].unique()))


# In[13]:


filter = (df['children']==0) & (df['adults']==0) & (df['babies']==0) 
df[filter]


# In[15]:


pd.set_option('display.max_column',32)
filter = (df['children']==0) & (df['adults']==0) & (df['babies']==0) 
df[filter]


# In[18]:


data=df[~filter]
data.head()


# In[20]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','No of guests']
country_wise_data


# In[21]:


get_ipython().system('pip install folium')


# In[25]:


import folium
from folium.plugins import HeatMap
folium.Map()
basemap=folium.Map()


# In[26]:


get_ipython().system('pip install plotly')


# In[40]:


import plotly.express as px
map_guest=px.choropleth(country_wise_data,
             locations=country_wise_data['country'],
             color=country_wise_data['No of guests'],
              hover_name=country_wise_data['country'],
              title='Home country of guests'
            )
map_guest.show()


# In[41]:


data.head()


# In[45]:


data2=data[data['is_canceled']==0]
data2.columns



# In[52]:


plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)

plt.title('Price of room types per night & per person')
plt.xlabel('Room type')
plt.ylabel('Price(Euro)')
plt.legend()
plt.show()


# In[57]:


data_resort=data[(data['hotel']=='Resort Hotel')& (data['is_canceled']==0)] 


# In[55]:


data_city=data[(data['hotel']=='City Hotel')& (data['is_canceled']==0)] 


# In[58]:


data_resort.head()


# In[60]:


resort_hotel=data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index
resort_hotel


# In[61]:


city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index
city_hotel


# In[65]:


final=resort_hotel.merge(city_hotel,on='arrival_date_month')


# In[66]:



get_ipython().system('pip install sorted-months-weekdays')


# In[73]:


get_ipython().system('pip install sort-dataframeby_monthorweek as sd')


# In[75]:


import sort_dataframeby_monthorweek as sd
def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)
sort_data(final,'month')
    


# In[ ]:


final.columns
px.line(final,x='month',y='',title = 'Room price per')


# In[76]:


data_resort.head()


# In[77]:


rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no of guests']
rush_resort


# In[78]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no of guests']
rush_city


# In[80]:


final_rush=rush_resort.merge(rush_city,on='month')
final_rush
final_rush.columns=['month','no of guests in resort','no of guest in city hotel']
final_rush


# In[82]:


final_rush=sort_data(final_rush,'month')
final_rush


# In[83]:


final_rush.columns


# In[84]:


px.line(final_rush,x='month',y=['no of guests in resort', 'no of guest in city hotel'],title='Total no of guests per months')


# In[85]:


data.head()


# In[86]:


data.corr()


# In[89]:


co_relation=data.corr()['is_canceled']
co_relation


# In[92]:


co_relation.abs().sort_values(ascending=False)


# In[93]:


data.groupby('is_canceled')['reservation_status'].value_counts()


# In[95]:


list_not=['days_in_waiting_list','arrival_date_year']


# In[101]:


num_features= [col for col in data.columns if data[col].dtype!='O' and col not in list_not]
num_features
#use this code bcoz this is optimize code 


# In[99]:


cols=[]
for col in data.columns:
    if data[col].dtype!='O' and col not in list_not:
        cols.append(col)
cols
    


# In[102]:


data.columns


# In[106]:


cat_not=['arrival_date_year','assigned_room_type','booking_changes', 'reservation_status','country','days_in_waiting_list',]
cat_not


# In[108]:


cat_features=[col for col in data.columns if data[col].dtype=='O' and col not in cat_not]


# In[109]:


cat_features


# In[110]:


data_cat=data[cat_features]
data_cat.head()


# In[111]:


data_cat.dtypes


# In[113]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')
data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])


# In[114]:


data_cat['year']=data_cat['reservation_status_date'].dt.year


# In[115]:


data_cat['month']=data_cat['reservation_status_date'].dt.month


# In[116]:


data_cat['day']=data_cat['reservation_status_date'].dt.day


# In[117]:


data_cat.head()


# In[126]:


data.dtypes
#data_cat.drop('reservation_status_date',axis=0,inplace=True)


# In[123]:


data_cat['cancellation']=data['is_canceled']


# In[127]:


data_cat.head()


# In[128]:


data_cat['market_segment'].unique()


# In[ ]:


###Mean Encoding


# In[131]:


cols=data_cat.columns[0:8]
cols


# In[133]:


data_cat.groupby(['hotel'])["cancellation"].mean()


# In[137]:


for col in cols:
    print(data_cat.groupby([col])["cancellation"].mean().to_dict())
    print('\n')
    


# In[138]:


for col in cols:
    dict=data_cat.groupby([col])["cancellation"].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict)


# In[139]:


data_cat.head()


# In[ ]:


###we convert all the string data to integer data beccause ml model cannot understand the string data


# In[144]:


dataframe = pd.concat([data_cat,data[num_features]],axis=1)


# In[145]:


dataframe.head()


# In[146]:


dataframe.drop('cancellation',axis=1,inplace=True)
dataframe.shape


# In[ ]:


###How to Handle Outliers(l16)


# In[147]:


dataframe.head()


# In[148]:


sns.distplot(dataframe['lead_time'])


# In[150]:


import numpy as np
def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[151]:


handle_outlier('lead_time')


# In[152]:


sns.distplot(dataframe['lead_time'])


# In[ ]:


##adr


# In[153]:


sns.distplot(dataframe['adr'])


# In[158]:


handle_outlier('adr')
sns.distplot(dataframe['adr'].dropna())


# In[ ]:


###Apply technique of feature importance on data to select most important features.(l17)


# In[159]:


dataframe.isnull().sum()


# In[161]:


dataframe.dropna(inplace=True)


# In[164]:





# In[165]:


y=dataframe['is_canceled']
x=dataframe.drop('is_canceled',axis=1)


# In[166]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[167]:


feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[168]:


feature_sel_model.fit(x,y)


# In[170]:


feature_sel_model.get_support()


# In[175]:


cols=x.columns
selected_feat=cols[feature_sel_model.get_support()]


# In[173]:


print('total_feature {}'.format(x.shape[1]))


# In[178]:


print('selected_features {}'.format(len(selected_feat)))


# In[179]:


selected_feat


# In[180]:


x=x[selected_feat]


# In[ ]:


###Our data is redy now its time to apply machine learning Algorithm and Cross validate our model


# In[182]:


from sklearn.model_selection import train_test_split


# In[183]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[ ]:


###Apply Logistic regression


# In[185]:


from sklearn.linear_model import LogisticRegression


# In[186]:


logreg=LogisticRegression()


# In[187]:


logreg.fit(X_train,y_train)


# In[189]:


y_pred=logreg.predict(X_test)
y_pred


# In[190]:


from sklearn.metrics import confusion_matrix


# In[191]:


confusion_matrix(y_test,y_pred)


# In[192]:


from sklearn.metrics import accuracy_score


# In[193]:


accuracy_score(y_test,y_pred)


# In[ ]:


###you guys think this is correct prediction but we have to cross validate the score


# In[194]:


from sklearn.model_selection import cross_val_score


# In[197]:


score =cross_val_score(logreg,x,y,cv=10)


# In[198]:


score.mean()


# In[ ]:


###THis mean  your 70% prediction going to correct.


# In[ ]:


###We are going to applay multiple algorithm in our data and check its accuracy


# In[ ]:


###we are going to cover 5 algorithm


# In[204]:


from sklearn.naive_bayes import GaussianNB


# In[200]:


from sklearn.linear_model import LogisticRegression


# In[201]:


from sklearn.neighbors import KNeighborsClassifier


# In[202]:


from sklearn.ensemble import RandomForestClassifier


# In[203]:


from sklearn.tree import DecisionTreeClassifier


# In[205]:


models=[]
models.append(('LogisticRegression',LogisticRegression()))
models.append(('Naive bayes',GaussianNB()))
models.append(('RandomForest',RandomForestClassifier()))
models.append(('Decision Tree',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))


# In[208]:


for name,model in models:
    print(name)
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(prediction,y_test))
    print('\n')
    print(accuracy_score(prediction,y_test))
    print('\n')
    


# In[ ]:





# In[ ]:





# In[ ]:





# End Of The Project#################################################################

# In[209]:


get_ipython().system('pip install pandoc')


# In[210]:


get_ipython().system('pip install Pyppeteer')

