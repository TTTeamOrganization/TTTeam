''' Recupérer données '''
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.linear_model import LinearRegression


Train_input = pd.read_csv('F:\MVA\SW\X_train.csv')
Test_input = pd.read_csv('F:\MVA\SW\X_test.csv')
Train_output = pd.read_csv('F:\MVA\SW\challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')

train_input = Train_input.copy()
test_input = Test_input.copy()
train_output = Train_output.copy()

''' Score '''
def score_function(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)/float(y_true.shape[0])
    
''' Traitement '''
# Traitement train_input
train_input = train_input.reindex_axis(sorted(train_input.columns), axis=1)
land_cols = ["hlres_1000","hlres_500","hlres_300", "hlres_100", "hlres_50", "hldres_1000","hldres_500","hldres_300", "hldres_100",
             "hldres_50", "industry_1000", "route_1000", "route_500", "route_300", "route_100", "port_5000", "natural_5000",
            "green_5000"]
train_input.loc[:, land_cols] = train_input.loc[:, land_cols].fillna( value=0)
train_input = pd.get_dummies(train_input, columns = ["pollutant"])

train_input["station_id"] = train_input["station_id"].astype('category')

#Remove, ne doit pas intervenir dans la prediction
train_input.drop(['station_id', 'ID'], axis=1, inplace=True)
test_input.drop(['station_id', 'ID'], axis=1, inplace=True)

train_input.describe()

''' Regression linéaire '''
# sample aléatoire pour la cross validation
rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

# Lin reg
linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))

''' Séparer les polluants '''

train_input_NO2 = train_input[train_input.pollutant_NO2 == 1].copy().reset_index()
train_input_PM10 = train_input[train_input.pollutant_PM10 == 1].copy().reset_index()
train_input_PM2_5 = train_input[train_input.pollutant_PM2_5 == 1].copy().reset_index()

train_output_NO2 = train_output[train_input.pollutant_NO2 == 1].copy().reset_index()
train_output_PM10 = train_output[train_input.pollutant_PM10 == 1].copy().reset_index()
train_output_PM2_5 = train_output[train_input.pollutant_PM2_5 == 1].copy().reset_index()


train_input_NO2.drop('index', axis=1, inplace=True)
train_input_PM10.drop('index', axis=1, inplace=True)
train_input_PM2_5.drop('index', axis=1, inplace=True)


''' 3 régression '''
rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
X = train_input_NO2.iloc[rindex,:]
Y = train_output_NO2.iloc[rindex].TARGET
X_test = train_input_NO2.iloc[~rindex,:]
Y_test = train_output_NO2.iloc[~rindex].TARGET
# Lin reg
linreg_NO2 = sk.linear_model.LinearRegression()
linreg_NO2.fit(X,Y)
square_error_NO2_train = (linreg_NO2.predict(X)-Y)**2
square_error_NO2 = (linreg_NO2.predict(X_test)-Y_test)**2
print("MSE test NO2 = "+str(np.mean(square_error_NO2)))

rindex = np.array(rd.sample(train_input_PM10.index, np.int(0.7*len(train_input_PM10))))
X = train_input_PM10.iloc[rindex,:]
Y = train_output_PM10.iloc[rindex].TARGET
X_test = train_input_PM10.iloc[~rindex,:]
Y_test = train_output_PM10.iloc[~rindex].TARGET
# Lin reg
linreg_PM10 = sk.linear_model.LinearRegression()
linreg_PM10.fit(X,Y)
square_error_PM10_train = (linreg_PM10.predict(X)-Y)**2
square_error_PM10 = (linreg_PM10.predict(X_test)-Y_test)**2
print("MSE test PM10 = "+str(np.mean(square_error_PM10)))

rindex = np.array(rd.sample(train_input_PM2_5.index, np.int(0.7*len(train_input_PM2_5))))
X = train_input_PM2_5.iloc[rindex,:]
Y = train_output_PM2_5.iloc[rindex].TARGET
X_test = train_input_PM2_5.iloc[~rindex,:]
Y_test = train_output_PM2_5.iloc[~rindex].TARGET
# Lin reg
linreg_PM2_5 = sk.linear_model.LinearRegression()
linreg_PM2_5.fit(X,Y)
square_error_PM2_5_train = (linreg_PM2_5.predict(X)-Y)**2
square_error_PM2_5 = (linreg_PM2_5.predict(X_test)-Y_test)**2
print("MSE test PM2_5 = "+str(np.mean(square_error_PM2_5)))

print("MSE train = "+str((np.sum(square_error_NO2_train)+np.sum(square_error_PM10_train)+np.sum(square_error_PM2_5_train))/(len(square_error_NO2_train) + len(square_error_PM10_train)+ len(square_error_PM2_5_train))))
print("MSE test = "+str((np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))))


''' Sur le test '''
# sur le test
res_NO2 = pd.DataFrame(linreg_NO2.predict(test_input_NO2.iloc[:,1:]), index = test_input_NO2.iloc[:,0], columns = ['TARGET'])
res_PM10 = pd.DataFrame(linreg_PM10.predict(test_input_PM10.iloc[:,1:]),  index = test_input_PM10.iloc[:,0], columns = ['TARGET'])
res_PM2_5 = pd.DataFrame(linreg_PM2_5.predict(test_input_PM2_5.iloc[:,1:]), index =  test_input_PM2_5.iloc[:,0], columns = ['TARGET'])
result = pd.concat([res_NO2, res_PM10, res_PM2_5]).sort_index()
result['ID'] = result.index
result = result[['ID', 'TARGET']]
result.to_csv("F:\MVA\SW\Outputs\challenge_output.csv" ,sep = ',', index=False)