import tabpy_client
from tabpy.tabpy_tools.client import Client
client = tabpy_client.Client('http://hbld-dmsapp01.allegistest.com:9004/')

def fraud_predictor5( _arg1, _arg2,_arg3):
    import pandas as pd
    row = {'shipping': _arg1,
           'shipping scheduled': _arg2,
          'country_str':_arg3}
    #Convert it into a dataframe
    test_data = pd.DataFrame(data = row,index=[0])
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    test_data['country_str']  = le.fit_transform(test_data['country_str'])
    #Predict the Fraud
    predprob_survival = random_forest.predict_proba(test_data)
    #Return only the probability
    return [probability[1] for probability in predprob_survival]

def late_delivery( _arg1, _arg2):
    import pandas as pd
    row = {'shipping scheduled': _arg1,
          'country_str':_arg2}
    #Convert it into a dataframe
    test_data = pd.DataFrame(data = row,index=[0])
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    test_data['country_str']  = le.fit_transform(test_data['country_str'])
    #Predict the late delivery probabilites
    predprob_late = random_forest_l.predict_proba(test_data)
    #Return only the probability
    return [probability[1] for probability in predprob_late]

#Deploying
client.deploy('fraud_predictor5', fraud_predictor5,'fraud_predictor probability',override = True)
client.deploy('late_delivery', late_delivery,'late_delivery_prop',override = True)
