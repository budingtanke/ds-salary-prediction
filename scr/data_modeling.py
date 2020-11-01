#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:53:45 2020

@author: Han
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import pickle

# import cleaned data
df = pd.read_csv('./data/glassdoor_cleaned.csv')
df.drop('company_text', axis=1, inplace=True)

# get dummy variables
df_dum = pd.get_dummies(df)
X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary']

# split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(sorted(list(X_test.index)))
y_test.sort_values()

################ Stats model OLS linear regression ################
# we get R-squared 0.608 and adjusted R-squared 0.531
X_sm =  sm.add_constant(X_train)
lr_sm = sm.OLS(y_train,X_sm).fit()
with open('./output/lr_summary.txt', 'w') as f:
    print(lr_sm.summary(), file=f)


############### sklearn linear regression #####################
lr = LinearRegression()


cv = KFold(n_splits=5, shuffle=True, random_state=7)
# in cross_val_socre, "scoring" is used to mesure performance, higher the better, but MAE, MSE are naturally descending scores (the smallest score is best), thus we need to use negative to invert the sorting. So this score for MAE, MSE will always be negative.

# get 22.11 MAE
np.mean(cross_val_score(lr, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= cv))

# lr.fit(X_train, y_train)
# print(lr.intercept_)
# print(lr.coef_)

################### sklearn lasso regression ################
# lasso base
lasso = Lasso()
cv = KFold(n_splits=5, shuffle=True, random_state=7)
# get 22.86 MAE
np.mean(cross_val_score(lasso,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 5))

# lasso with standardized numeric features and cross validation, find best alpha
kf = KFold(n_splits=5, shuffle=True, random_state=7)
numeric_features = ['Rating', 'age', 'description']


alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    cv_score = []
    for train_index, test_index in kf.split(X_train, y_train):
        train, test = X_train.iloc[train_index,], X_train.iloc[test_index,]
        train_label, test_label = y_train.iloc[train_index,], y_train.iloc[test_index,]
        scaler = StandardScaler().fit(train[numeric_features])
        train_transformed = train.copy()
        train_transformed.loc[:,numeric_features] = scaler.transform(train[numeric_features])
        lasso = Lasso(alpha=i/100)
        lasso.fit(train_transformed, train_label)
        test_transformed = test.copy()
        test_transformed.loc[:, numeric_features] = scaler.transform(test[numeric_features])
        score = mean_absolute_error(test_label, lasso.predict(test_transformed))
        cv_score.append(score)
    error.append(np.mean(cv_score))

# plot alpha vs MAE
plt.plot(alpha,error)
plt.xlabel('alpha')
plt.ylabel('MAE')
plt.title('lasso regression alpha vs MAE')
plt.savefig('output/lasso_alpha.png')

# best alpha: 0.21, best MAE: 20.59
df_err = pd.DataFrame({'alpha': alpha, 'MAE': error})
df_err[df_err['MAE']==min(df_err['MAE'])]

# fit training data to see coef
lasso = Lasso(alpha=0.21)
scaler = StandardScaler()
X_train_transformed = X_train.copy()
X_train_transformed[numeric_features] = scaler.fit_transform(X_train[numeric_features])
lasso.fit(X_train_transformed, y_train)

# export coefficients to csv
# some interesting coefficients:
# Rating: 3.68
# age: 2.44
# size 100000+: -4.35 ?
# size 5001 to 10000: 19.4
# sector finance: 11.36
# sector_Information Technology: 7.7
# sector_Retail: 34.17
# revenue_$10+ billion: 15.5
# state_CA: 34.23
# state_CT: 10.7
# state_MA: 12.9
# state_NY: 23.5
# seniority_director: 3.3
lasso_coef = pd.DataFrame({'features': X_train.columns, 'coef': lasso.coef_})
lasso_coef.to_csv('./output/lasso_coef.csv')

# 70 features out of 120 have 0 coefficients
sum(lasso.coef_==0)


###################### sklearn random forest ####################
# random forest base
rf = RandomForestRegressor()
cv = KFold(n_splits=5, shuffle=True, random_state=7)
# get MAE 17.43
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= cv))

parameters = {'n_estimators':range(10,300,50), \
              'criterion':('mse','mae'), \
                  'max_features':('auto','sqrt','log2')}
gs_rf = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=cv)

gs_rf.fit(X_train,y_train)

# best MAE 17.01
gs_rf.best_score_
# criterion='mse', n_estimators = 260, max_features='sqrt'
gs_rf.best_estimator_


################# test ######################
best_lr = LinearRegression()
best_lr.fit(X_train, y_train)
predict_lr = best_lr.predict(X_test)

best_lasso = Lasso(alpha=0.21)
best_lasso.fit(X_train, y_train)
predict_lasso = best_lasso.predict(X_test)

best_rf = RandomForestRegressor(criterion='mse', n_estimators = 260, max_features='sqrt')
best_rf.fit(X_train, y_train)
predict_rf = best_rf.predict(X_test)

print(mean_absolute_error(y_test,predict_lr))
print(mean_absolute_error(y_test,predict_lasso))
print(mean_absolute_error(y_test,predict_rf))


################ store model and test value for Flask API ###############


with open('./scr/FlaskAPI/models/best_rf.p', 'wb') as f:
    pickle.dump(best_rf, f)
    
    
with open('./scr/FlaskAPI/models/best_rf.p', 'rb') as f:
    model = pickle.load(f)
    
test_value = list(X_test.iloc[0,:].values)
model.predict(np.array(test_value).reshape(1,-1))[0]
with open('models/test_value.p', 'wb') as f:
    pickle.dump(test_value, f)
    
with open('models/test_value.p', 'rb') as f:
    X = pickle.load(f)
    

model.predict(X)
