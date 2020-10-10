import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
dataset=pd.read_csv("DataCoSupplyChainDataset.csv",header= 0,encoding= 'unicode_escape')

# Adding first name and last name together to create new column
dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str)+dataset['Customer Lname'].astype(str)
data=dataset.drop(['Customer Email','Product Status','Customer Password','Customer Street','Customer Fname','Customer Lname',
           'Latitude','Longitude','Product Description','Product Image','Order Zipcode','shipping date (DateOrders)'],axis=1)
data['Customer Zipcode']=data['Customer Zipcode'].fillna(0)
def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range
lower_range,upper_range=outlier_treatment(data['Product Price'])
data.drop(data[(data['Product Price'] < lower_range) | (data['Product Price'] > upper_range)].index , inplace=True)
train_data=data.copy()
train_data['fraud'] = np.where(train_data['Order Status'] == 'SUSPECTED_FRAUD', 1, 0)
train_data['late_delivery']=np.where(train_data['Delivery Status'] == 'Late delivery', 1, 0)
#Dropping columns with repeated values
train_data.drop(['Delivery Status','Late_delivery_risk','Order Status', 'order date (DateOrders)'], axis=1, inplace=True)

le = preprocessing.LabelEncoder()
train_data['Order Country']  = le.fit_transform(train_data['Order Country'])
train_data['Order State']    = le.fit_transform(train_data['Order State'])
Xf=train_data[['Days for shipping (real)','Days for shipment (scheduled)','Order Country']]
yf=train_data['fraud']
train_x,test_x,train_y,test_y = train_test_split(Xf,yf,test_size = 0.2, random_state = 42)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y.values.ravel())
random_forest.score(train_x, train_y)