import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.ensemble import GradientBoostingRegressor

''' Grad Boost simple '''
grad_boost = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

# sample aléatoire pour la cross validation
rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))
X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

# Lin reg

grad_boost.fit(X,Y)

print('MSE train = ' +str(score_function(Y, grad_boost.predict(X))))
print('MSE test = ' +str(score_function(Y_test, grad_boost.predict(X_test))))

''' Triple Grad Boost '''
rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
X = train_input_NO2.iloc[rindex,:]
Y = train_output_NO2.iloc[rindex].TARGET
X_test = train_input_NO2.iloc[~rindex,:]
Y_test = train_output_NO2.iloc[~rindex].TARGET

# grad boost
grad_boost_NO2 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

grad_boost_NO2.fit(X,Y)
square_error_NO2_train = (grad_boost_NO2.predict(X)-Y)**2
square_error_NO2 = (grad_boost_NO2.predict(X_test)-Y_test)**2
print("MSE test NO2 = "+str(np.mean(square_error_NO2)))

rindex = np.array(rd.sample(train_input_PM10.index, np.int(0.7*len(train_input_PM10))))
X = train_input_PM10.iloc[rindex,:]
Y = train_output_PM10.iloc[rindex].TARGET
X_test = train_input_PM10.iloc[~rindex,:]
Y_test = train_output_PM10.iloc[~rindex].TARGET

# grad boost
grad_boost_PM10 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
grad_boost_PM10.fit(X,Y)
square_error_PM10_train = (grad_boost_PM10.predict(X)-Y)**2
square_error_PM10 = (grad_boost_PM10.predict(X_test)-Y_test)**2
print("MSE test PM10 = "+str(np.mean(square_error_PM10)))

rindex = np.array(rd.sample(train_input_PM2_5.index, np.int(0.7*len(train_input_PM2_5))))
X = train_input_PM2_5.iloc[rindex,:]
Y = train_output_PM2_5.iloc[rindex].TARGET
X_test = train_input_PM2_5.iloc[~rindex,:]
Y_test = train_output_PM2_5.iloc[~rindex].TARGET

# grad boost
grad_boost_PM2_5 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

grad_boost_PM2_5.fit(X,Y)
square_error_PM2_5_train = (grad_boost_PM2_5.predict(X)-Y)**2
square_error_PM2_5 = (grad_boost_PM2_5.predict(X_test)-Y_test)**2
print("MSE test PM2_5 = "+str(np.mean(square_error_PM2_5)))

print("MSE train = "+str((np.sum(square_error_NO2_train)+np.sum(square_error_PM10_train)+np.sum(square_error_PM2_5_train))/(len(square_error_NO2_train) + len(square_error_PM10_train)+ len(square_error_PM2_5_train))))
print("MSE test = "+str((np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))))

# sur le test
res_NO2 = pd.DataFrame(grad_boost_NO2.predict(test_input_NO2.iloc[:,1:]), index = test_input_NO2.iloc[:,0], columns = ['TARGET'])
res_PM10 = pd.DataFrame(grad_boost_PM10.predict(test_input_PM10.iloc[:,1:]),  index = test_input_PM10.iloc[:,0], columns = ['TARGET'])
res_PM2_5 = pd.DataFrame(grad_boost_PM2_5.predict(test_input_PM2_5.iloc[:,1:]), index =  test_input_PM2_5.iloc[:,0], columns = ['TARGET'])
result = pd.concat([res_NO2, res_PM10, res_PM2_5]).sort_index()
result['ID'] = result.index
result = result[['ID', 'TARGET']]
result.to_csv("F:\MVA\SW\Outputs\challenge_output_V2.csv" ,sep = ',', index=False)
