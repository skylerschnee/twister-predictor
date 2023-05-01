#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('figure', figsize=(13, 7))
plt.rc('font', size=16)
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt 




def mod_tts(df_by_year):
    '''
    mod_tts takes in df_by_year and splits into train, validate, test
    '''
    train_size = int(len(df_by_year) * .5)
    validate_size = int(len(df_by_year) * .3)
    test_size = int(len(df_by_year) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df_by_year[: train_size]
    validate = df_by_year[train_size : validate_end_index]
    test = df_by_year[validate_end_index : ]
    
    return train, validate, test



def evaluate(target_var, validate, yhat_df):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_and_eval(target_var, train, validate, yhat_df):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, validate, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()


def get_simple_average(train, validate):
    '''
    produce simple average predicitons
    '''
    # compute simple average of casualties (from train data)
    avg_cas = round(train['casualties'].mean(), 2)

    yhat_df = pd.DataFrame({'casualties':[avg_cas]}, index=validate.index)

    return yhat_df


# function to store rmse for comparison purposes
def append_eval_df(model_type, target_var, validate, yhat_df, eval_df):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)




def get_lov(train, validate):
    '''
    produce last observed value predicitons
    '''
    # take the last item of casualties and assign to variable
    last_cas = train['casualties'][-1:][0]

    yhat_df = pd.DataFrame(
        {'casualties': [last_cas]}, index=validate.index)
    
    return yhat_df




def get_hlt_results(train, validate, yhat_df):
    '''
    produce Holt's linear trend predicitons
    '''
    
    model = Holt(train['casualties'], exponential=False, damped=True)
    model = model.fit(optimized=True)
    yhat_values = model.predict(start = validate.index[0],
                                  end = validate.index[-1])
    yhat_df['casualties'] = round(yhat_values, 2)
    
    return yhat_df, model


def get_hst_results(train):
    '''
    get_hst_results takes in train data, fits and creates 8 different HST models,
    then returns a df containg the results.
    '''
    # create and fit
    hst_cas_fit1 = ExponentialSmoothing(train.casualties, seasonal_periods=3, trend='add', seasonal='add').fit()
    hst_cas_fit2 = ExponentialSmoothing(train.casualties, seasonal_periods=3, trend='add', seasonal='mul').fit()
    hst_cas_fit3 = ExponentialSmoothing(train.casualties, seasonal_periods=3, trend='add', seasonal='add', damped=True).fit()
    hst_cas_fit4 = ExponentialSmoothing(train.casualties, seasonal_periods=3, trend='add', seasonal='mul', damped=True).fit()
    hst_cas_fit5 = ExponentialSmoothing(train.casualties, seasonal_periods=10, trend='add', seasonal='add').fit()
    hst_cas_fit6 = ExponentialSmoothing(train.casualties, seasonal_periods=10, trend='add', seasonal='mul').fit()
    hst_cas_fit7 = ExponentialSmoothing(train.casualties, seasonal_periods=10, trend='add', seasonal='add', damped=True).fit()
    hst_cas_fit8 = ExponentialSmoothing(train.casualties, seasonal_periods=10, trend='add', seasonal='mul', damped=True).fit()
    
    # create df for results and fill with sse values
    results_quantity=pd.DataFrame({'model':['hst_cas_fit1', 'hst_cas_fit2',
                                            'hst_cas_fit3', 'hst_cas_fit4',
                                            'hst_cas_fit5', 'hst_cas_fit6',
                                            'hst_cas_fit7', 'hst_cas_fit8'],
                                  'SSE':[hst_cas_fit1.sse, hst_cas_fit2.sse,
                                         hst_cas_fit3.sse, hst_cas_fit4.sse,
                                         hst_cas_fit5.sse, hst_cas_fit6.sse,
                                         hst_cas_fit7.sse, hst_cas_fit8.sse]})
    return results_quantity.sort_values(by='SSE'), hst_cas_fit8


def final_plot(target_var, train, validate, test, yhat_df):
    '''
    produce graph visualizing test results
    '''
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], color='#377eb8', label='train')
    plt.plot(validate[target_var], color='#ff7f00', label='validate')
    plt.plot(test[target_var], color='#4daf4a',label='test')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.legend()
    plt.title(target_var)
    plt.show()
    
    
    
def get_test_results(target_var, train, validate, test, yhat_df):
    '''
    get_test_results will calculate RMSE and visualize the predicitons
    '''
    # calculate rmse
    rmse_cas_total = sqrt(mean_squared_error(test['casualties'], 
                                           yhat_df['casualties']))

    print('FINAL PERFORMANCE OF MODEL ON TEST DATA')
    print('rmse- casualty total: ', rmse_cas_total)

    final_plot(target_var, train, validate, test, yhat_df)

    
def get_mod_test_reslts(target_var, train, validate, test, yhat_df):
    '''
    get_mod_test_reslts will produce rmse and visualzation after modifying the test data to remove 
    the outlier of the 2011 super outbreak. We reset the casualty value for this year to the median 
    of the subset
    '''
    # set outlier to equal median
    yhat_df.loc['2011-12-31', 'casualties'] = yhat_df.casualties.median()
    test.loc['2011-12-31', 'casualties'] = test.casualties.median()
    # calculate RMSE
    rmse_cas_total = sqrt(mean_squared_error(test['casualties'], 
                                           yhat_df['casualties']))

    print('FINAL PERFORMANCE OF MODEL ON MODIFIED TEST DATA')
    print('rmse- casualty total: ', rmse_cas_total)

    final_plot(target_var, train, validate, test, yhat_df)

